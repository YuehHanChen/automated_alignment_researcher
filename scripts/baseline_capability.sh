#!/bin/bash
#SBATCH --job-name=capbase
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/aar/work
#
# Measure the CAPABILITY basket (mmlu, gsm8k, ifeval) PER MODEL. These are the
# capability_filter floors — they must be per-model, because the gate is "don't
# regress vs THIS model's own capability" (a fixed qwen floor is too lax for a
# stronger 7B and too strict for gemma-2b). Rule-scored, NO judge.
#
# Prep ONCE on the login node (publishes the capability data to the scratch suite):
#   R=/opt/aar/aar_repo
#   PY=/opt/aar/work/git/python
#   PYTHONPATH=$R $PY $R/scripts/publish_suite.py --suite capability \
#       --holdout-dir /opt/aar/work/aar_repo_runs/_capbaseline
# Then sweep all 6 models (one GPU each, all parallel):
#   sbatch --array=0-5 scripts/baseline_capability.sh
#   single: sbatch scripts/baseline_capability.sh <hf-model-id>
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
SCRATCH=/opt/aar/work/aar_repo_runs/_capbaseline

export PYTHONPATH="${R}"
export HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# SAME decoding the eval worker uses to score capability (parity): greedy, AUTO budget
# capped at 4096, no no_repeat guard. A drift here would corrupt the floor.
export EVAL_GPUS=auto
export EVAL_AUTO_CEILING=4096
export EVAL_BATCH_SIZE=8
unset EVAL_MAX_NEW_TOKENS
# Rule-scored basket — no judge model is invoked.

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
else
  MODEL="${1:?usage: sbatch [--array=0-5] baseline_capability.sh [<hf-model-id>]}"
fi
TAG="${MODEL//\//_}"
cd "${R}"
echo "[capbase] $(date) model=${MODEL}"
${PY} -m aar.eval_pod.run_eval \
  --suite "${SCRATCH}/capability/capability.yaml" \
  --model "${MODEL}" \
  --secret-dir "${SCRATCH}/capability" \
  --out "${SCRATCH}/cap_${TAG}.json"
echo "[capbase] result for ${MODEL}:"
cat "${SCRATCH}/cap_${TAG}.json"
echo; echo "=== DONE ${MODEL} ==="
