#!/bin/bash
#SBATCH --job-name=aar-evaljob
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00   # 2h (was 1h): privacy's rule-scored legs (confaide/privaci_bench/gdpr) run at
                          # ceiling 4096 / no-ngram BY BASELINE DESIGN (parity-locked — can't cap without
                          # re-baselining), so 7 legs of legit generation overran the old 1h wall and got
                          # TIMEOUT-killed before writing the composite -> no score -> no finding. Other axes
                          # finish well under 1h, so this only RAISES their ceiling (they still exit early).
#SBATCH --output=/opt/aar/eval-user/eval_job_%j.out
#
# EPHEMERAL EVAL — scores ONE submission and exits. SUBMIT THIS AS eval-user.
# It is the per-submission counterpart of eval_worker.sh's inner loop: it loads the
# staged model, scores it against the LOCKED holdout, and writes the held-out-STRIPPED
# composite to SCORES_DIR. Spawned by eval_watcher.sh (one per submission) on --qos=low
# (uncapped; short job, so preemption just means the watcher resubmits this one rid).
# The original eval_worker.sh DAEMON is left intact as the fallback.
#
# Args: <run_id> [suite].  All dirs are env-honoring (default = the gemma queue/holdout),
# so a per-model team passes SUBMISSIONS_DIR/SCORES_DIR/HOLDOUT_DIR via the watcher's env.
set -uo pipefail
RID="${1:?usage: eval_job.sh <run_id> [suite]}"
SUITE="${2:-refusal}"
REPO=/opt/aar/aar_repo                 # research-owned code, world-readable
# judge_deps holds tiktoken+sentencepiece+blobfile for the refusal PAPER judges — same as eval_worker.sh.
export PYTHONPATH="${REPO}:/opt/aar/work/judge_deps"
export HF_HOME="${HF_HOME:-/opt/aar/eval-user/hf}"
export HF_TOKEN="${HF_TOKEN:-$(grep -m1 '^HF_TOKEN=' /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HARNESS_TRANSPORT=fs
export HOLDOUT_DIR="${HOLDOUT_DIR:-/opt/aar/eval-user/holdout}"
export HELDOUT_SCORES_DIR="${HELDOUT_SCORES_DIR:-${HOLDOUT_DIR}/heldout_scores}"
export SUBMISSIONS_DIR="${SUBMISSIONS_DIR:-/opt/aar/work/aar_repo_runs/submissions}"
export SCORES_DIR="${SCORES_DIR:-/opt/aar/work/aar_repo_runs/scores}"
export OAI_API="$(grep -h '^OAI_API=' /opt/aar/eval-user/.oai_env /opt/aar/eval-user/.env 2>/dev/null | head -1 | cut -d= -f2-)"
# Anthropic key for the JUDGE_BACKEND=anthropic judges (honesty: mask + deceptionbench, claude-haiku-4-5).
# Load whichever name is present (the judge's _anthropic_key() checks all three). KEY=VALUE extraction
# only (no `source` — the .env has a $(...) line that would abort).
for _ak in ANTHROPIC_API_KEY ANT_high_prio_API ANT_API_KEY; do
  _av="$(grep -m1 "^${_ak}=" /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
  [ -n "${_av}" ] && export "${_ak}=${_av}"
done
# EVAL_GPUS=auto => run_eval uses however many GPUs SLURM gave us (set by --gres at submit).
export EVAL_GPUS="${EVAL_GPUS:-auto}"
# DECODING PARITY (critical): the trained-model eval MUST use the SAME decoding as the
# suite's baseline (benchmark_docs/<suite>/baseline.json) or the composite delta is
# corrupted — greedy is NOT batch-invariant, and the AUTO budget is capped by the ceiling.
# models.py defaults (batch 16, ceiling 2048) DIVERGE from the sycophancy/refusal baselines
# (batch 8, ceiling 4096). Pin them so eval == baseline. greedy + no_repeat_ngram=0 already
# match. NB: keep these in lock-step with benchmark_docs/<suite>/baseline.json's `decoding`.
case "${SUITE}" in
  sycophancy|refusal) export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}";  export EVAL_AUTO_CEILING="${EVAL_AUTO_CEILING:-4096}" ;;
  honesty)            export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"; export EVAL_AUTO_CEILING="${EVAL_AUTO_CEILING:-1024}"; export EVAL_NO_REPEAT_NGRAM="${EVAL_NO_REPEAT_NGRAM:-4}" ;;
  power_seeking)      export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"; export EVAL_AUTO_CEILING="${EVAL_AUTO_CEILING:-1024}"; export EVAL_NO_REPEAT_NGRAM="${EVAL_NO_REPEAT_NGRAM:-4}" ;;  # baseline.json: sample T=1 batch 32; machiavelli legs are logprob (batch-invariant), instrumental_eval freeform
  bias)               export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"; export EVAL_AUTO_CEILING="${EVAL_AUTO_CEILING:-4096}" ;;  # 2026-07-02: KEEP 32 (do NOT raise to 64). Verified bs=64 BREAKS bias_race_content parity: baseline 0.45 (bs<=32, matches running evals ~0.47) -> 0.18 at bs=64 (left-padding artifact on the SHORT story prompts; letters/bios/scenes are batch-stable). Speedup comes from gpu:6 parallelism (watcher), which is parity-safe. Batch is pinned so a stray env can't push it to 64 and miscalibrate the composite.
  hallucination|faithfulness)      export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
                      # ragtruth FAITHFULNESS must use the finetuned Llama-2-13b detector (~0.80 F1),
                      # not the prompt-judge fallback (~0.40 F1) — the baseline (benchmark_docs) was
                      # measured WITH it, so omitting it scores trained models on a DIFFERENT, worse
                      # scorer and the trained-baseline delta is invalid. Pin both (mirror baseline_hallucination.sh).
                      export RAGTRUTH_DETECTOR="${RAGTRUTH_DETECTOR:-/opt/aar/work/aar_repo_runs/_ragtruth_detector}"
                      export RAGTRUTH_DETECTOR_BASE="${RAGTRUTH_DETECTOR_BASE:-meta-llama/Llama-2-13b-hf}" ;;
esac
# JUDGE backend MUST match the axis's baseline (eval_worker.sh sets this too; it was MISSING here
# -> hallucination would default to openai but is baselined on local Qwen). Pin it per axis:
#   hallucination/faithfulness/privacy = LOCAL Qwen2.5-7B ; honesty = ANTHROPIC claude-haiku-4-5
#   (2026-06-21, cheaper than gpt-4o + no OpenAI; re-baselined with it) ; sycophancy/refusal = OpenAI.
case "${SUITE}" in
  hallucination) export JUDGE_BACKEND="${JUDGE_BACKEND:-local}"; export JUDGE_MODEL_LOCAL="${JUDGE_MODEL_LOCAL:-Qwen/Qwen2.5-7B-Instruct}" ;;
  # faithfulness (2026-06-27 redesign): the 2 MULTI-TURN free-form legs (faith_mt_grounded/claimcheck) + ragtruth
  # are Haiku-judged free-form → ANTHROPIC claude-haiku-4-5 @ conc 100 (mirror reward_hacking/bias). ragtruth's
  # FAITHFULNESS still uses the finetuned detector (RAGTRUTH_DETECTOR set in the batch-size case above); only its
  # utility gate moves local-Qwen → Haiku (hence ragtruth is re-baselined on Haiku). logprob legs (llm_aggrefact_A/B,
  # summedits) have no judge → unaffected.
  faithfulness) export JUDGE_BACKEND="${JUDGE_BACKEND:-anthropic}"; export JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"; export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-100}"; export ANTHROPIC_MIN_INTERVAL_S="${ANTHROPIC_MIN_INTERVAL_S:-0}" ;;
  # privacy: privacylens is the only judged leg (confaide + privaci_bench are rule-scored). Switched to
  # ANTHROPIC claude-haiku-4-5 (2026-06-22) — a stronger CoT judge than local Qwen for the leakage + 0-3
  # helpfulness judgments. The Anthropic key is sourced above (same plumbing as honesty).
  privacy) export JUDGE_BACKEND="${JUDGE_BACKEND:-anthropic}"; export JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"; export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-8}"; export ANTHROPIC_MIN_INTERVAL_S="${ANTHROPIC_MIN_INTERVAL_S:-0.1}" ;;   # 8 / 0.1s (was 4 / 0.2s): 0 live 429s = headroom; speeds the privacylens judged leg (throughput-only, doesn't touch decoding parity). anthropic_chat still honors Retry-After.
  # power_seeking: machiavelli_power/harm + machiavelli_heldout are RULE-scored (logprob, no judge); only
  # instrumental_eval is judged and uses the LOCAL Qwen2.5-7B 3-way judge (CONV/NOCONV/REFUSAL) — pin it.
  power_seeking) export JUDGE_BACKEND="${JUDGE_BACKEND:-local}"; export JUDGE_MODEL_LOCAL="${JUDGE_MODEL_LOCAL:-Qwen/Qwen2.5-7B-Instruct}" ;;
  # concealing_uncertainty: llm_uncertainty_bench + verbalized_ece are rule/logprob-scored (no judge); only
  # abstentionbench is judged, on the LOCAL Qwen2.5-7B BLIND abstain-detector — pin it to match the baseline.
  concealing_uncertainty) export JUDGE_BACKEND="${JUDGE_BACKEND:-local}"; export JUDGE_MODEL_LOCAL="${JUDGE_MODEL_LOCAL:-Qwen/Qwen2.5-7B-Instruct}" ;;
  # reward_hacking: the MCQ legs (revealing_score/world_affecting_reward) are RULE-scored (logprob, no judge);
  # the JUDGE legs are reward_hacking_freeform + reward_harm_freeform (2026-06-30 de-enum) + rh_mt_reward
  # (transcript). All call the ANTHROPIC claude-haiku-4-5 3-way judge (HACK/HONOR/EXCLUDE) directly. Setting the
  # backend here is AXIS-LEVEL, so it covers every judge leg incl the new reward_harm_freeform. The Anthropic
  # key is sourced above (same plumbing as honesty).
  reward_hacking) export JUDGE_BACKEND="${JUDGE_BACKEND:-anthropic}"; export JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"; export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-100}"; export ANTHROPIC_MIN_INTERVAL_S="${ANTHROPIC_MIN_INTERVAL_S:-0}" ;;   # conc 100 / no min-interval throttle: the 3 judge legs (~174 Haiku calls/eval) judge in ~2 waves; high-prio key + anthropic_chat honors Retry-After on any 429.
  honesty) export JUDGE_BACKEND="${JUDGE_BACKEND:-anthropic}"; export JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"; export MASK_JUDGE_MODEL="${MASK_JUDGE_MODEL:-claude-haiku-4-5}" ;;
  # bias (2026-06-28): the 2 MULTI-TURN FREE-FORM legs (bias_mt_decision/bias_mt_occupation) are judged by
  # VERDICT EXTRACTION (engagement + sign) → ANTHROPIC claude-haiku-4-5 @ conc 100 (mirror reward_hacking/
  # faithfulness). bbq + bbq_heldout are RULE-scored (logprob, no judge). WITHOUT this case the bias eval falls
  # through, the Haiku judge is never set, the MT judge_fn is None, every pair skips → n=0 (the 06-28 bug).
  bias) export JUDGE_BACKEND="${JUDGE_BACKEND:-anthropic}"; export JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"; export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-100}"; export ANTHROPIC_MIN_INTERVAL_S="${ANTHROPIC_MIN_INTERVAL_S:-0}" ;;
  sycophancy|refusal) export JUDGE_BACKEND="${JUDGE_BACKEND:-openai}" ;;
esac
PY=/opt/aar/work/git/python  # world-readable venv
cd "${REPO}"

echo "[evaljob $(hostname)] scoring ${RID} (suite ${SUITE}) -> ${SCORES_DIR}"
if [ -f "${SCORES_DIR}/${RID}.json" ]; then echo "[evaljob] already scored ${RID}; nothing to do"; echo "=== DONE ==="; exit 0; fi
if PYTHONUNBUFFERED=1 ${PY} -u -m aar.eval_pod.entrypoint --run-id "${RID}" --suite "${SUITE}"; then
  echo "[evaljob] done ${RID}"; echo "=== DONE ==="; exit 0
else
  echo "[evaljob] FAILED ${RID}"; exit 1
fi
