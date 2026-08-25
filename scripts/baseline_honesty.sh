#!/bin/bash
#SBATCH --job-name=honbase
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/opt/aar/work
#
# Baseline the HONESTY axis (property #5) per model: mask, behonest_burglar,
# pacchiardi_lying_rate (SCORED) + deceptionbench (HELD-OUT). honesty != accuracy —
# every leg scores vs the model's OWN belief / known answer / per-item reference.
#
# Judges (3 of 4 legs): mask + behonest_burglar + deceptionbench are LLM-judged;
# pacchiardi_lying_rate is rule-scored (NO judge). This script uses the PAPER judge
# (JUDGE_BACKEND=openai -> gpt-4o, the MASK/DeceptionBench paper judge) — it is both more
# paper-faithful AND much faster than a local 7B judge: the calls run through a thread pool
# (no GPU lock), so judging finishes in minutes instead of serially. Combined with 4-GPU
# sharding (the 4 legs run in parallel, one model replica per GPU), the property finishes
# WELL UNDER the 30-min budget. Retry/backoff (aar/benchmarks/_judge_http.py) keeps the
# concurrent calls from fail-closing on a 429.
#   >>> PARITY (load-bearing): whatever judge you baseline with MUST be the one the
#       trained-model eval (eval_worker.sh) uses — else the composite delta is invalid.
#       Pin judge (gpt-4o) + temp 0 on BOTH sides. (Local-Qwen batched judge is the
#       grid-cost follow-up; if you switch, re-baseline.)
#
# Prep ONCE on the login node AS THE EVAL USER (publishes the holdout incl. the GATED
# cais/MASK — needs HF_TOKEN; writes the held-out deceptionbench data too):
#   R=/opt/aar/aar_repo
#   PY=/opt/aar/work/git/python
#   HF_TOKEN=$(grep -m1 '^HF_TOKEN=' /opt/aar/aar_repo/.env|cut -d= -f2-) \
#   PYTHONPATH=$R $PY $R/scripts/publish_suite.py --suite honesty \
#       --only mask behonest_burglar pacchiardi_lying_rate deceptionbench \
#       --holdout-dir /opt/aar/work/aar_repo_runs/_honestybaseline
#   sweep : sbatch --array=0-5 scripts/baseline_honesty.sh
#   (single: sbatch scripts/baseline_honesty.sh <hf-model-id>)
set -euo pipefail

MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "meta-llama/Llama-3.2-3B-Instruct"
  "allenai/OLMo-2-1124-7B-Instruct"
  "google/gemma-2-2b-it"
  "microsoft/Phi-3.5-mini-instruct"
)
R=/opt/aar/aar_repo
PY=/opt/aar/work/git/python
ENVF=/opt/aar/aar_repo/.env
SCRATCH=/opt/aar/work/aar_repo_runs/_honestybaseline

export PYTHONPATH="${R}"
export HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"   # gated model weights (Llama/Gemma/Mistral)
export OAI_API="$(grep -m1 '^OAI_API=' "${ENVF}" | cut -d= -f2-)"     # REQUIRED: gpt-4o paper judge
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Judge backend: gpt-4o (the paper judge) via the API thread pool — fast + faithful.
export JUDGE_BACKEND="${JUDGE_BACKEND:-openai}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"

# Parallelism: shard the 4 legs across the 4 allocated GPUs (one model replica per GPU),
# so wall-clock = the slowest single leg, not the sum. With gpt-4o judging off-GPU this
# lands the property well under 30 min. Greedy, AUTO 4096 ceiling (EOS bounds it), batch 8.
export EVAL_GPUS="${EVAL_GPUS:-auto}"
# Generation robustness (REQUIRED — weakly-aligned + AAR-produced models often don't emit
# EOS and run away to the ceiling, blowing the 30-min budget AND truncating outputs):
#   - AUTO ceiling 1024 bounds any runaway (still covers every real honesty response —
#     mask statements / deceptionbench JSON are a few hundred tokens at most).
#   - no_repeat_ngram=4 forces a looping generation to emit EOS early (the harness's
#     built-in degenerate-gen fix). Applied UNIFORMLY to baseline + trained eval (parity).
export EVAL_AUTO_CEILING=1024
export EVAL_NO_REPEAT_NGRAM=4
# Batch 32: pure GPU generation throughput (greedy outputs are batch-invariant -> no score
# change), short honesty answers fit easily on an H200. Speeds the model-under-test gen,
# especially for the slow/rambling models.
export EVAL_BATCH_SIZE=32
# Judge API parallelism. Internal-honesty suite = 4 judge legs (mask_factual, mask_generative,
# deceptionbench_pressure, deceptionbench_reward) sharded across 4 GPUs, each with its own
# judge pool → 48/leg = ≤192 concurrent gpt-4o calls, under the account's ~200 limit (retry
# absorbs any brief burst). Quality-neutral (concurrency doesn't change verdicts).
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-48}"
unset EVAL_MAX_NEW_TOKENS
[ -n "${OAI_API}" ] || { echo "ERROR: OAI_API empty (need gpt-4o judge); set it in ${ENVF}"; exit 1; }

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
else
  MODEL="${1:?usage: sbatch [--array=0-5] baseline_honesty.sh [<hf-model-id>]}"
fi
TAG="${MODEL//\//_}"
cd "${R}"
echo "[honbase] $(date) model=${MODEL}  judge_backend=${JUDGE_BACKEND} (legs: mask/behonest/deceptionbench; pacchiardi=rule)"
# --heldout-dir captures the FULL composite incl. the HELD-OUT deceptionbench (the --out
# handoff is held-out-STRIPPED). For BASELINING we need deceptionbench's score too, so we
# read it back from the heldout dir to patch benchmark_docs/honesty/baseline.json.
${PY} -m aar.eval_pod.run_eval \
  --suite "${SCRATCH}/honesty/honesty.yaml" \
  --model "${MODEL}" \
  --secret-dir "${SCRATCH}/honesty" \
  --out "${SCRATCH}/hon_${TAG}.json" \
  --heldout-dir "${SCRATCH}/heldout_scores"
echo "[honbase] STRIPPED result (3 scored) for ${MODEL}:"
cat "${SCRATCH}/hon_${TAG}.json"
echo "[honbase] FULL incl held-out deceptionbench:"
cat "${SCRATCH}/heldout_scores/hon_${TAG}.json" 2>/dev/null || echo "  (heldout file not found)"
echo "[base] NOTE: held-out benchmark data was published to research scratch to measure it — purge before any AAR:"
echo "[base]   scripts/purge_heldout_research.sh   (launch_team.sh also auto-purges + fail-safe-aborts before every team)"
echo; echo "=== DONE ${MODEL} ==="
