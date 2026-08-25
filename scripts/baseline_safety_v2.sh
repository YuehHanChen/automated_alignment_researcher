#!/bin/bash
#SBATCH --job-name=sycbase
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/aar/work
#
# Baseline the NEW 3-benchmark sycophancy safety suite (sycophancy_eval + sycon_fp
# + elephant_aita) on one base model. Reads the data PRE-PUBLISHED to the scratch
# suite (publish_suite.py was run once on the login node), so this job needs no
# dataset internet — only model weights (HF_TOKEN set; downloads if uncached) and
# the gpt-4o judge (OAI_API). One GPU per model so all 6 models schedule in
# parallel; the 3 benchmarks run sequentially (judge work is I/O-bound + threaded,
# so a single GPU is not the bottleneck).
# Usage:
#   single : sbatch scripts/baseline_safety_v2.sh <hf-model-id>
#   sweep  : sbatch --array=0-5 scripts/baseline_safety_v2.sh   (one model per task)
set -euo pipefail

# The 6-model sweep (plan.md). In array mode, SLURM_ARRAY_TASK_ID indexes this.
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
SCRATCH=/opt/aar/work/aar_repo_runs/_safetybaseline_v2

export PYTHONPATH="${R}"
export HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export OAI_API="$(grep -m1 '^OAI_API=' "${ENVF}" | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_GPUS=auto
# No artificial truncation of real answers: AUTO budget, ceiling 4096 — 3-8x longer
# than any legitimate response to these prompts (factual answer / false-premise
# rebuttal / AITA verdict+reasoning are all well under ~1.5k tokens). Fully-
# unbounded context is NOT used because the batch KV-cache (batch x max_tokens)
# would OOM the GPU on a degenerate sequence. Smaller batch keeps the cache safe
# when a generation does run long. Anything still hitting 4096 = model degeneration
# (repetition loop), surfaced by _note_truncation, not a real answer being cut.
export EVAL_AUTO_CEILING=4096
export EVAL_BATCH_SIZE=8
unset EVAL_MAX_NEW_TOKENS
# NO decoding guard. An A/B (scripts/ab_ngram.sh) showed `no_repeat_ngram` shifts
# the score (size-3 −40%, size-8 −8%) AND can *increase* truncation (forcing
# no-repeats makes a model ramble longer into the budget) — strictly worse. At
# ceiling 4096 with plain greedy decoding, truncation is already ~0; any residual
# is genuine model degeneration on that item, harmless to our scorers (the signal
# — verdict / challenge / answer — is at the START of the response) and reported
# as a count rather than hidden by altering decoding. EVAL_NO_REPEAT_NGRAM stays 0.
export JUDGE_BACKEND=openai
export JUDGE_MODEL=gpt-4o

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
else
  MODEL="${1:?usage: sbatch [--array=0-5] baseline_safety_v2.sh [<hf-model-id>]}"
fi
TAG="${MODEL//\//_}"
cd "${R}"
echo "[sycbase] $(date) model=${MODEL}"
${PY} -m aar.eval_pod.run_eval \
  --suite "${SCRATCH}/sycophancy/sycophancy.yaml" \
  --model "${MODEL}" \
  --secret-dir "${SCRATCH}/sycophancy" \
  --out "${SCRATCH}/baseline_${TAG}.json"
echo "[sycbase] result for ${MODEL}:"
cat "${SCRATCH}/baseline_${TAG}.json"
echo "[base] NOTE: held-out benchmark data was published to research scratch to measure it — purge before any AAR:"
echo "[base]   scripts/purge_heldout_research.sh   (launch_team.sh also auto-purges + fail-safe-aborts before every team)"
echo; echo "=== DONE ${MODEL} ==="
