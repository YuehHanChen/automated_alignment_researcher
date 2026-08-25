#!/bin/bash
#SBATCH --job-name=refbase
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=/opt/aar/work
#
# Baseline the 4 harmful-request-REFUSAL benchmarks (strongreject, harmbench, jbb,
# jbb_artifacts) per model. All judge-scored by a LOCAL Qwen2.5-7B grid judge (loaded on
# the same GPU as the model under test). Decoding matches the eval worker (greedy, AUTO
# budget cap 4096, no guard) for parity; harmbench pins gen=512 inside the plugin.
#
# Prep ONCE on the login node (publishes the 4 refusal benchmarks' data to the scratch suite):
#   R=/opt/aar/aar_repo
#   PY=/opt/aar/work/git/python
#   HF_TOKEN=$(grep -m1 '^HF_TOKEN=' /opt/aar/aar_repo/.env|cut -d= -f2-) \
#   PYTHONPATH=$R $PY $R/scripts/publish_suite.py --suite refusal \
#       --only strongreject harmbench jbb jbb_artifacts \
#       --holdout-dir /opt/aar/work/aar_repo_runs/_refusalbaseline
#   sweep : sbatch --array=0-5 scripts/baseline_refusal.sh
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
SCRATCH=/opt/aar/work/aar_repo_runs/_refusalbaseline

# judge_deps holds tiktoken+blobfile (needed by the Llama-3 / HarmBench judge tokenizers;
# the shared .venv is read-only so they're installed to a side dir on PYTHONPATH).
export PYTHONPATH="${R}:/opt/aar/work/judge_deps"
export HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"   # gated Llama-3 / Llama-Guard judges
export OAI_API="$(grep -m1 '^OAI_API=' "${ENVF}" | cut -d= -f2-)"     # GPT-4o-mini StrongREJECT rubric judge
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Parity with the eval worker: greedy, AUTO budget cap 4096, no no_repeat guard, batch 8.
# Sequential benchmarks (EVAL_GPUS=1) so the model under test + the 3 local PAPER judges
# (HarmBench-13b-cls, Llama-Guard-3-8B, Llama-3-8B) all co-reside on the one 143GB H200.
export EVAL_GPUS=1
export EVAL_AUTO_CEILING=4096
export EVAL_BATCH_SIZE=8
unset EVAL_MAX_NEW_TOKENS
export REFUSAL_JUDGE_DEVICE=cuda

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
else
  MODEL="${1:?usage: sbatch [--array=0-5] baseline_refusal.sh [<hf-model-id>]}"
fi
TAG="${MODEL//\//_}"
cd "${R}"
echo "[refbase] $(date) model=${MODEL}  judges=paper (strongreject:gpt-4o-mini, harmbench:HarmBench-13b-cls, jbb:Llama-Guard-3-8B+Llama-3-8B)"
${PY} -m aar.eval_pod.run_eval \
  --suite "${SCRATCH}/refusal/refusal.yaml" \
  --model "${MODEL}" \
  --secret-dir "${SCRATCH}/refusal" \
  --out "${SCRATCH}/ref_${TAG}.json"
echo "[refbase] result for ${MODEL}:"
cat "${SCRATCH}/ref_${TAG}.json"
echo "[base] NOTE: held-out benchmark data was published to research scratch to measure it — purge before any AAR:"
echo "[base]   scripts/purge_heldout_research.sh   (launch_team.sh also auto-purges + fail-safe-aborts before every team)"
echo; echo "=== DONE ${MODEL} ==="
