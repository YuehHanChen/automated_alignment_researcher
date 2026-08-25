#!/bin/bash
#SBATCH --job-name=aar-eval-worker
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/opt/aar/eval-user/eval_worker_%j.out
#
# Eval worker — SUBMIT THIS AS eval-user (the secret-holding user):
#   ssh eval-user@<login> 'cd <repo>; sbatch scripts/eval_worker.sh'
#
# Holds one GPU and drains the submission queue: for each run the research user
# stages a model for (marker `.submitted`, no scores yet), it scores the model
# against the LOCKED holdout (which only this user can read) and writes the
# composite to the shared SCORES_DIR. The research/AAR side never reads the
# holdout — it only stages a model and polls SCORES_DIR. Exits after being idle.

set -uo pipefail
REPO=/opt/aar/aar_repo          # research-owned code, world-readable
# judge_deps holds tiktoken+sentencepiece+blobfile for the refusal-property PAPER judges
# (HarmBench-13b-cls / Llama-Guard-3-8B / Llama-3-8B tokenizers). MUST match the baseline
# runner (scripts/baseline_refusal.sh) or refusal scores fail-closed -> "safe" and break the
# baseline/method parity. NB: the judge MODEL weights must also be in this user's HF_HOME
# (or point HF_HOME at the shared cache) — pre-download them eval-side once.
export PYTHONPATH="${REPO}:/opt/aar/work/judge_deps"
export HF_HOME="${HF_HOME:-/opt/aar/eval-user/hf}"   # eval-owned cache (overridable)
# Gated judges/models (Llama-Guard-3, Llama-3, HarmBench-cls, gemma, ...) need an HF token —
# sourced from the eval-user .env (sycophancy's local judges didn't, so this was unset before).
export HF_TOKEN="${HF_TOKEN:-$(grep -m1 '^HF_TOKEN=' /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Generation budget: leave EVAL_MAX_NEW_TOKENS UNSET so scoring uses the per-model
# AUTO budget (models.py: model's remaining context, capped by EVAL_AUTO_CEILING).
# This is the SAME default the baseline run uses — baseline and method-scoring can
# no longer diverge (the 256-vs-512 truncation bug that broke the gsm8k floor).
export HARNESS_TRANSPORT=fs
# All four dirs are env-honoring (default = today's path). A per-model worker for a
# SECOND team on the same axis passes model-namespaced dirs (holdout_<model>,
# submissions_<model>, scores_<model>) so two models' eval queues + baselines never
# cross. Unset (the normal single-team case) => byte-identical to before.
# HOLDOUT_DIR default is DEFERRED: it's set per-MODEL (holdout/<model_tag>) in the per-team block
# below, derived from the team id so it ALWAYS matches what publish_holdout wrote — two same-axis/
# different-model teams can't collide. Explicit HOLDOUT_DIR (or the legacy fallback at the end) wins.
export HOLDOUT_DIR="${HOLDOUT_DIR-}"
# GOLDEN per-benchmark decoding source — EVAL-SIDE ONLY. benchmark_docs names the held-out
# benchmarks, so it must NEVER live on the research (aar-user) side or the AAR could read
# which benchmark is held out. The eval pod (this user) reads it to score each benchmark at
# the exact config its baseline used. Lives next to the holdout, mode-700.
export BENCHMARK_DOCS_DIR="${BENCHMARK_DOCS_DIR:-/opt/aar/eval-user/benchmark_docs}"
# HELD-OUT scores: per-team (set in the per-team block); legacy fallback finalized after HOLDOUT_DIR.
export SUBMISSIONS_DIR="${SUBMISSIONS_DIR:-/opt/aar/work/aar_repo_runs/submissions}"
export SCORES_DIR="${SCORES_DIR:-/opt/aar/work/aar_repo_runs/scores}"      # research-readable handoff (stripped)
export OAI_API="$(grep -h '^OAI_API=' /opt/aar/eval-user/.oai_env /opt/aar/eval-user/.env 2>/dev/null | head -1 | cut -d= -f2-)"
# Anthropic key for JUDGE_BACKEND=anthropic (honesty: mask + deceptionbench, claude-haiku-4-5).
for _ak in ANTHROPIC_API_KEY ANT_high_prio_API ANT_API_KEY; do
  _av="$(grep -m1 "^${_ak}=" /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
  [ -n "${_av}" ] && export "${_ak}=${_av}"
done
PY=/opt/aar/work/git/python  # world-readable venv
SUITE="${1:-sycophancy}"
IDLE_EXIT="${2:-1800}"   # exit after this many seconds with no work
# Benchmarks score IN PARALLEL, one per allocated GPU. EVAL_GPUS=auto (default)
# means run_eval uses however many GPUs SLURM gave us — so the parallelism is set
# by --gres at submit time, which scripts/launch_eval_worker.sh sizes to the
# suite's benchmark count (2-3 safety + 3 capability). The #SBATCH --gres above
# is just a fallback when eval_worker.sh is sbatch'd directly; the launcher
# overrides it with --gres=gpu:<n_benchmarks>.
export EVAL_GPUS="${3:-auto}"
# BATCH-SIZE PARITY: greedy decode is NOT batch-invariant (FP-path drift ~0.5 pt), so the
# trained-eval batch MUST equal the suite's baseline batch or the composite delta is corrupted.
# hallucination was baselined at batch 32 (3.2x faster gen on H200, ~36/140 GB). Other suites
# default to models.py's 16 here — NOTE their baselines used 8 (pre-existing mismatch; see
# train_baseline_sync.md). sycophancy IS pinned-correct (batch 8 + ceiling 4096, matching
# benchmark_docs/sycophancy/baseline.json — keep in lock-step with eval_job.sh's same case).
case "${SUITE}" in
  sycophancy|refusal) export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}";  export EVAL_AUTO_CEILING="${EVAL_AUTO_CEILING:-4096}" ;;
  honesty)            export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"; export EVAL_AUTO_CEILING="${EVAL_AUTO_CEILING:-1024}"; export EVAL_NO_REPEAT_NGRAM="${EVAL_NO_REPEAT_NGRAM:-4}" ;;
  hallucination|faithfulness) export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
                      # ragtruth FAITHFULNESS needs the finetuned Llama-2-13b detector (~0.80 F1), not the
                      # prompt-judge fallback — pin both (mirror eval_job.sh / baseline_hallucination.sh).
                      export RAGTRUTH_DETECTOR="${RAGTRUTH_DETECTOR:-/opt/aar/work/aar_repo_runs/_ragtruth_detector}"
                      export RAGTRUTH_DETECTOR_BASE="${RAGTRUTH_DETECTOR_BASE:-meta-llama/Llama-2-13b-hf}" ;;
  power_seeking)      export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}" ;;
esac
# NB: the above are a FALLBACK only — run_eval now pins decoding PER-BENCHMARK from the
# eval-side benchmark_docs/<group>/baseline.json (the capability basket at its own config on
# every axis), so each benchmark is scored exactly as its baseline was.
# JUDGE backend MUST also match the axis's baseline, or judge-scored benchmarks diverge from
# their baseline. The baseline scripts pin it per axis: hallucination = LOCAL Qwen2.5-7B
# (baseline_hallucination.sh); honesty/sycophancy/refusal = OpenAI per-paper (baseline_honesty/
# newsyc/safety_v2). eval_worker had set NONE -> run_eval defaulted to openai -> hallucination
# would have been judged by OpenAI but baselined on local Qwen. Pin it to match:
case "${SUITE}" in
  # hallucination: judge-free (logprob) legs except ragtruth's finetuned detector — local Qwen, pin both sides.
  hallucination) export JUDGE_BACKEND="${JUDGE_BACKEND:-local}"; export JUDGE_MODEL_LOCAL="${JUDGE_MODEL_LOCAL:-Qwen/Qwen2.5-7B-Instruct}" ;;
  # faithfulness (2026-06-27 redesign): the 2 MULTI-TURN free-form legs (faith_mt_grounded/claimcheck) AND
  # ragtruth are now Haiku-judged free-form generation → ANTHROPIC claude-haiku-4-5 @ conc 100 (mirrors the
  # `bias)` case; the logprob legs llm_aggrefact_A/B + summedits have no judge → unaffected). NB: this moves
  # ragtruth's judge from local-Qwen → Haiku, so ragtruth must be re-baselined on Haiku (baseline_faithfulness.sh).
  faithfulness) export JUDGE_BACKEND="${JUDGE_BACKEND:-anthropic}"; export JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"; export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-100}" ;;
  # privacy: privacylens is the only judged leg (confaide + privaci_bench are rule-scored). Switched to
  # ANTHROPIC claude-haiku-4-5 (2026-06-22) — a stronger CoT judge than local Qwen for the leakage + 0-3
  # helpfulness judgments. The Anthropic key is sourced above (same plumbing as honesty).
  privacy) export JUDGE_BACKEND="${JUDGE_BACKEND:-anthropic}"; export JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"; export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"; export ANTHROPIC_MIN_INTERVAL_S="${ANTHROPIC_MIN_INTERVAL_S:-0.2}" ;;   # throttle Haiku to stay under the 429 limit (anthropic_chat also honors Retry-After)
  # power_seeking: machiavelli legs are rule-scored (logprob); only instrumental_eval is judged, on the
  # LOCAL Qwen2.5-7B 3-way judge (CONV/NOCONV/REFUSAL) — pin it both sides.
  power_seeking) export JUDGE_BACKEND="${JUDGE_BACKEND:-local}"; export JUDGE_MODEL_LOCAL="${JUDGE_MODEL_LOCAL:-Qwen/Qwen2.5-7B-Instruct}" ;;
  # concealing_uncertainty: llm_uncertainty_bench + verbalized_ece are rule/logprob-scored (no judge); only
  # abstentionbench is judged, on the LOCAL Qwen2.5-7B BLIND abstain-detector — pin it to match the baseline.
  concealing_uncertainty) export JUDGE_BACKEND="${JUDGE_BACKEND:-local}"; export JUDGE_MODEL_LOCAL="${JUDGE_MODEL_LOCAL:-Qwen/Qwen2.5-7B-Instruct}" ;;
  # reward_hacking: the MCQ legs (loophole_*/revealing_score) are RULE-scored (logprob, no judge); only the
  # FREE-FORM leg (reward_hacking_freeform[_heldout]) is judged. Its plugin calls the ANTHROPIC claude-haiku-4-5
  # 3-way judge (HACK/HONOR/EXCLUDE) directly; pin backend + concurrency/throttle (mirror privacy + eval_job.sh)
  # so the Haiku calls are fast and rate-limit-safe. The Anthropic key is sourced above (same plumbing as honesty).
  reward_hacking) export JUDGE_BACKEND="${JUDGE_BACKEND:-anthropic}"; export JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"; export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-100}"; export ANTHROPIC_MIN_INTERVAL_S="${ANTHROPIC_MIN_INTERVAL_S:-0}" ;;   # conc 100 / no throttle (2026-06-27; mirror eval_job.sh): the 3 free-form legs judge in 1-2 waves; high-prio key + Retry-After backoff on any 429.
  honesty) export JUDGE_BACKEND="${JUDGE_BACKEND:-anthropic}"; export JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"; export MASK_JUDGE_MODEL="${MASK_JUDGE_MODEL:-claude-haiku-4-5}" ;;
  # bias: the two MULTI-TURN FREE-FORM legs (bias_mt_decision/bias_mt_occupation) are judged for engagement ×
  # A-vs-B invariance; bbq + bbq_heldout are RULE-scored (logprob, no judge). ANTHROPIC claude-haiku-4-5 — a
  # strong CoT judge for the free-form decision/recommendation comparisons. Concurrency 100 (Haiku handles it;
  # anthropic_chat honors Retry-After on any 429 bursts). Anthropic key sourced above (same plumbing as honesty).
  bias) export JUDGE_BACKEND="${JUDGE_BACKEND:-anthropic}"; export JUDGE_MODEL="${JUDGE_MODEL:-claude-haiku-4-5}"; export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-100}" ;;
  sycophancy|refusal) export JUDGE_BACKEND="${JUDGE_BACKEND:-openai}" ;;
esac
# Run SEVERAL of these concurrently to parallelize scoring (one worker is a
# bottleneck for many chains — ~10-14 min/submission). As the eval user:
#   for i in $(seq 3); do sbatch scripts/eval_worker.sh sycophancy 8000; done
# The atomic mkdir-claim below lets N workers drain the same queue without
# double-scoring: exactly one worker claims each submission.
WORKER_ID="${SLURM_JOB_ID:-$$}"
# EVAL-SIDE per-team organization (mirror of the research TEAM_DIR). The research
# SUBMISSIONS_DIR is .../aar_teams/<TEAM_ID>/submissions, so derive <TEAM_ID> and keep
# THIS team's eval-PRIVATE artifacts under the EVAL user's OWN aar_teams/<TEAM_ID>/
# (mode-700): the held-out (generalization) scores, the per-run eval work staging, and
# this worker's log. The locked HOLDOUT itself (test set + baselines) stays per-axis-model
# (shared, comparable across teams) — only the per-RUN RESULTS move into the team folder.
case "${SUBMISSIONS_DIR}" in
  */aar_teams/*/submissions)
    EVAL_TEAM_ID="$(basename "$(dirname "${SUBMISSIONS_DIR}")")"
    EVAL_TEAMS="/opt/aar/work"
    EVAL_TEAM_DIR="${EVAL_TEAMS}/${EVAL_TEAM_ID}"
    mkdir -p "${EVAL_TEAM_DIR}/heldout_scores" "${EVAL_TEAM_DIR}/_evalwork"
    # mode-700 all the way down: held-out (generalization) scores are eval-SECRET; the
    # research/AAR user must never read them.
    chmod 700 "${EVAL_TEAMS}" "${EVAL_TEAM_DIR}" "${EVAL_TEAM_DIR}/heldout_scores" "${EVAL_TEAM_DIR}/_evalwork" 2>/dev/null || true
    export HELDOUT_SCORES_DIR="${EVAL_TEAM_DIR}/heldout_scores"   # per-team (was shared HOLDOUT_DIR/heldout_scores)
    export HARNESS_RUNS_DIR="${EVAL_TEAM_DIR}"                    # entrypoint stages _evalwork/<rid> under here
    # PER-MODEL HOLDOUT (robust, automatic). The team id is <axis>-<model>-<YYYYMMDD>-<HHMMSS>[-<n>];
    # strip the axis prefix + the timestamp suffix to recover <model_tag> — the IDENTICAL tag
    # publish_holdout used (so a same-axis/different-model team reads ITS OWN holdout/<model>/<axis>,
    # never a sibling's). Robust to hyphenated model ids (sed strips only the timestamp).
    _AXIS="${EVAL_TEAM_ID%%-*}"
    _MTAG="$(printf '%s' "${EVAL_TEAM_ID#${_AXIS}-}" | sed -E 's/-[0-9]{8}-[0-9]{6}(-[0-9]+)?$//')"
    export HOLDOUT_DIR="${HOLDOUT_DIR:-/opt/aar/work"
    exec >> "${EVAL_TEAM_DIR}/eval_worker_${WORKER_ID}.out" 2>&1  # route this worker's log into the team folder
    echo "[worker ${WORKER_ID}] eval-side team folder: ${EVAL_TEAM_DIR}"
    echo "[worker ${WORKER_ID}] per-model HOLDOUT_DIR=${HOLDOUT_DIR} (model_tag=${_MTAG}, axis=${_AXIS})"
    ;;
esac
# Finalize: legacy flat fallback if still unset (non-per-team run), then HELD-OUT scores default.
export HOLDOUT_DIR="${HOLDOUT_DIR:-/opt/aar/eval-user/holdout}"
export HELDOUT_SCORES_DIR="${HELDOUT_SCORES_DIR:-${HOLDOUT_DIR}/heldout_scores}"
CLAIMS="${SCORES_DIR}/.claims"
mkdir -p "${CLAIMS}" 2>/dev/null || true

cd "${REPO}"
echo "[worker ${WORKER_ID}] draining submissions for suite=${SUITE} (idle-exit ${IDLE_EXIT}s)"
last_work=$(date +%s)
while true; do
  did=0
  for d in "${SUBMISSIONS_DIR}"/*/; do
    rid=$(basename "$d")
    [ -f "${d}.submitted" ] || continue
    [ -f "${SCORES_DIR}/${rid}.json" ] && continue          # already scored
    # Atomic claim: exactly one worker's mkdir succeeds for a given rid, so
    # parallel workers never double-score. (Claims live in the eval-writable
    # SCORES_DIR so every worker — same OS user — can create them.)
    mkdir "${CLAIMS}/${rid}" 2>/dev/null || continue        # claimed by another worker
    echo "[worker ${WORKER_ID}] scoring ${rid}"
    if PYTHONUNBUFFERED=1 ${PY} -u -m aar.eval_pod.entrypoint --run-id "${rid}" --suite "${SUITE}"; then
      echo "[worker ${WORKER_ID}] done ${rid}"
    else
      echo "[worker ${WORKER_ID}] FAILED ${rid}"
      rmdir "${CLAIMS}/${rid}" 2>/dev/null || true          # release so another worker can retry
    fi
    did=1; last_work=$(date +%s)
  done
  if [ "$did" = 0 ]; then
    now=$(date +%s)
    [ $(( now - last_work )) -ge "${IDLE_EXIT}" ] && { echo "[worker ${WORKER_ID}] idle ${IDLE_EXIT}s, exiting"; break; }
    sleep 20
  fi
done
echo "=== DONE ==="
