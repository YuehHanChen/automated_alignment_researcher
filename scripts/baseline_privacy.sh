#!/bin/bash
#SBATCH --job-name=privbase
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# (was 200G — for the OLD on-GPU local Qwen judge. With JUDGE_BACKEND=anthropic the judge is an API
# call, so only the <=7B model-under-test is loaded; 64G is ample and matches eval_job.sh. The 200G
# request was failing cgroup setup ("Device or resource busy") on busy nodes; 64G schedules cleanly.)
#SBATCH --time=04:00:00
#SBATCH --output=/opt/aar/work
#
# Baseline the PRIVACY benchmarks per model (property #7). IN PROGRESS — 3 safety legs:
#   confaide      (rule; combined tier-2b Pearson + tier-3 leak + tier-4 utility-gated)
#   privaci_bench (rule; 3-way regulatory CI-compliance, macro-F1 + logprob fallback)
#   privacylens   (JUDGE; agentic-trajectory leakage, headline = P(helpful>=2 AND not-leak),
#                  LOCAL Qwen2.5-7B judge for leakage + helpfulness)
# HELD-OUT: privaci_gdpr_heldout (GDPR domain; same macro-F1 compliance scorer as privaci_bench). The
# privacy suite now INCLUDES it (role=held_out): run_eval STRIPS it from --out and writes its FULL score
# to --heldout-dir (HELDOUT_SCORES_DIR, eval-private). Read that to patch baseline.json, THEN PURGE.
# Decoding MUST match the eval worker
# (sample/temp-1/top_p-1/seed-1234/batch-32; rule-scored legs use the global ceiling) and the
# privacylens JUDGE MODEL must match both sides — see benchmark_docs/privacy/train_baseline_sync.md.
#
# Prep ONCE on the login node. privacylens needs the agent prompts PRE-RENDERED in the pl_venv
# (procoder/toolemu) — see aar/benchmarks/privacylens/_render.py:
#   R=/opt/aar/aar_repo
#   PY=/opt/aar/work/git/python
#   export PRIVACYLENS_RENDERED=/opt/aar/work/bench_src/privacylens_rendered.json
#   PYTHONPATH=$R $PY $R/scripts/publish_suite.py --suite privacy --only privacylens \
#       --holdout-dir /opt/aar/work/aar_repo_runs/_privbaseline
#   (confaide + privaci_bench already baselined; add them to --only to re-measure.)
#   sweep : sbatch --array=0-4 scripts/baseline_privacy.sh
set -euo pipefail

# The current 5 OS target models (benchmark_docs/README.md, refreshed 2026-06-08).
MODELS=(
  "Qwen/Qwen3.5-2B"
  "meta-llama/Llama-3.2-3B-Instruct"
  "allenai/Olmo-3-7B-Instruct"
  "microsoft/Phi-4-mini-instruct"
  "google/gemma-2-2b-it"
)
R=/opt/aar/aar_repo
PY=/opt/aar/work/git/python
# Path overrides so this can ALSO run as the EVAL user (on qos=high32 — the 32-GPU pool — to dodge the
# research-side GPU contention + cgroup churn). Override ENVF (eval .env: anthropic + HF keys), SCRATCH
# (eval-writable), HF_HOME (eval cache). Defaults = the research-side paths.
ENVF="${ENVF:-/opt/aar/aar_repo/.env}"
SCRATCH="${SCRATCH:-/opt/aar/work/aar_repo_runs/_privbaseline}"

export PYTHONPATH="${R}"
export HF_HOME="${HF_HOME:-/opt/aar/work/hf_cache}"
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" 2>/dev/null | cut -d= -f2- || true)"
# Anthropic key for the privacylens JUDGE_BACKEND=anthropic judge (claude-haiku-4-5) — load whichever
# name is present (the judge's _anthropic_key() checks all three) so the BASELINE judge matches the eval
# worker (parity). KEY=VALUE extraction only (no `source` — the .env has a $(...) line).
for _ak in ANTHROPIC_API_KEY ANT_high_prio_API ANT_API_KEY; do
  # `|| true` is REQUIRED: under `set -euo pipefail`, grep finding nothing (the first key name is
  # usually absent) exits 1 -> pipefail -> the command-sub fails -> set -e kills the whole script
  # before it ever runs (manifests as an instant "FAILED" + a cgroup-cleanup line, NOT a node issue).
  _av="$(grep -m1 "^${_ak}=" "${ENVF}" 2>/dev/null | cut -d= -f2- || true)"
  [ -n "${_av}" ] && export "${_ak}=${_av}" || true
done
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Parity with the eval worker. confaide + privaci_bench are rule-scored (no judge); privacylens uses
# the ANTHROPIC claude-haiku-4-5 CoT judge (2026-06-22; make_anthropic_judge_text_batch) — pin the SAME
# judge here as in eval_worker.sh's `privacy)` case so the baseline <-> per-iteration eval parity holds.
# (The judge is now an API call, not on-GPU; the model under test still uses the gpu:1 for generation.)
export EVAL_GPUS=auto
export EVAL_AUTO_CEILING=4096
export EVAL_BATCH_SIZE=32
unset EVAL_MAX_NEW_TOKENS
export JUDGE_BACKEND=anthropic
export JUDGE_MODEL=claude-haiku-4-5
# HELD-OUT (privaci_gdpr_heldout): run_eval writes its FULL score here and STRIPS it from --out.
# Run-as-eval-user → point this at the eval-private mode-700 dir; run-in-research-scratch (below) →
# purge_heldout_research.sh removes it afterward. Either way the GDPR baseline ends up ONLY in the
# eval-side benchmark_docs/privacy/baseline.json (mode-700).
export HELDOUT_SCORES_DIR="${SCRATCH}/heldout_scores"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
else
  MODEL="${1:?usage: sbatch [--array=0-4] baseline_privacy.sh [<hf-model-id>]}"
fi
TAG="${MODEL//\//_}"
cd "${R}"
echo "[privbase] $(date) model=${MODEL}  (privacy suite; privacylens = local Qwen2.5-7B judge)"
${PY} -m aar.eval_pod.run_eval \
  --suite "${SCRATCH}/privacy/privacy.yaml" \
  --model "${MODEL}" \
  --secret-dir "${SCRATCH}/privacy" \
  --out "${SCRATCH}/priv_${TAG}.json" \
  --heldout-dir "${HELDOUT_SCORES_DIR}"
echo "[privbase] result for ${MODEL} (research handoff, held-out STRIPPED):"
cat "${SCRATCH}/priv_${TAG}.json"
echo "[privbase] held-out (privaci_gdpr_heldout) FULL score, eval-private:"
cat "${HELDOUT_SCORES_DIR}/${TAG}.json" 2>/dev/null || cat "${HELDOUT_SCORES_DIR}"/*.json 2>/dev/null || echo "  (no heldout file — check --heldout-dir)"
echo; echo "=== DONE ${MODEL} ==="
# After the sweep: patch the 5 per-model means + 95% CIs into benchmark_docs/privacy/baseline.json
# (regenerate from these JSONs; do NOT hand-edit numbers) — the safety legs from priv_*.json and the
# HELD-OUT (privaci_gdpr_heldout) from ${HELDOUT_SCORES_DIR}/*.json. THEN run, as the research user:
#   scripts/purge_heldout_research.sh   (launch_team.sh also auto-purges + fail-safe-aborts before every team)
# so no GDPR held-out data/score remains research-readable.
