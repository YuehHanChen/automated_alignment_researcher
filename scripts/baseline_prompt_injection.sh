#!/bin/bash
#SBATCH --job-name=pibase
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/aar/work
#
# Baseline the prompt-injection (property #2) benchmarks per model. All RULE-scored (c),
# NO judge. Decoding matches the eval worker (greedy, AUTO budget cap 4096, no guard, batch
# 8) for parity. Each Tensor Trust benchmark runs 2 arms (attack + access-code/DV).
#
# Prep ONCE on the login node (publishes the data to the scratch suite):
#   R=/opt/aar/aar_repo
#   PY=/opt/aar/work/git/python
#   PYTHONPATH=$R HF_HOME=/opt/aar/work/hf_cache $PY $R/scripts/publish_suite.py \
#       --suite prompt_injection \
#       --only injecagent open_prompt_injection tensor_trust_hijack tensor_trust_extract \
#       --holdout-dir /opt/aar/work/aar_repo_runs/_pibaseline
#   sweep : sbatch --array=0-5 scripts/baseline_prompt_injection.sh
#   single: sbatch scripts/baseline_prompt_injection.sh <hf-model-id>
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
SCRATCH=/opt/aar/work/aar_repo_runs/_pibaseline

export PYTHONPATH="${R}"
export HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Parity with the eval worker: greedy, AUTO budget cap 4096, no no_repeat guard, batch 8.
export EVAL_GPUS=auto
export EVAL_AUTO_CEILING=256   # PI outputs are short; only the start matters (access-granted / label / first Action)
export EVAL_BATCH_SIZE=8
unset EVAL_MAX_NEW_TOKENS

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
else
  MODEL="${1:?usage: sbatch [--array=0-5] baseline_prompt_injection.sh [<hf-model-id>]}"
fi
TAG="${MODEL//\//_}"
cd "${R}"
# open_prompt_injection is the pi HELD-OUT: run_eval STRIPS it from --out and writes its
# full score only to --heldout-dir. Without --heldout-dir it is dropped — so the pi held-out
# baseline was never captured. Pass it (matches baseline_hallucination.sh).
export HELDOUT_SCORES_DIR="${SCRATCH}/heldout_scores"
echo "[pibase] $(date) model=${MODEL}"
${PY} -m aar.eval_pod.run_eval \
  --suite "${SCRATCH}/prompt_injection/prompt_injection.yaml" \
  --model "${MODEL}" \
  --secret-dir "${SCRATCH}/prompt_injection" \
  --out "${SCRATCH}/pi_${TAG}.json" \
  --heldout-dir "${HELDOUT_SCORES_DIR}"
echo "[pibase] result for ${MODEL}:"
cat "${SCRATCH}/pi_${TAG}.json"
echo "[base] NOTE: held-out benchmark data was published to research scratch to measure it — purge before any AAR:"
echo "[base]   scripts/purge_heldout_research.sh   (launch_team.sh also auto-purges + fail-safe-aborts before every team)"
echo; echo "=== DONE ${MODEL} ==="
