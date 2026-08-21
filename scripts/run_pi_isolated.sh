#!/bin/bash
#SBATCH --job-name=piISO
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/aar/work
set -euo pipefail
# Prompt-injection model set: Llama-3.2-3B + Gemma-2-2B DROPPED (Llama ~10% injecagent valid-rate;
# Gemma's chat template rejects the system role). The 4 below support the system role natively.
MODELS=("Qwen/Qwen2.5-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" \
        "allenai/OLMo-2-1124-7B-Instruct" "microsoft/Phi-3.5-mini-instruct")
R=/opt/aar/work/git/_aar_pi_isolated          # ISOLATED (sync can't clobber it)
PY=/opt/aar/work/git/python
ENVF=/opt/aar/aar_harness/.env
SCRATCH=/opt/aar/work/aar_harness_runs/_pibaseline
export PYTHONPATH="${R}" HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" EVAL_GPUS=auto EVAL_BATCH_SIZE=8
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"; TAG="${MODEL//\//_}"; cd "${R}"
echo "[piISO] $(date) model=${MODEL} repo=${R}"
${PY} -m aar.eval_pod.run_eval --suite "${SCRATCH}/prompt_injection/prompt_injection.yaml" \
  --model "${MODEL}" --secret-dir "${SCRATCH}/prompt_injection" --out "${SCRATCH}/piISO_${TAG}.json"
echo "[piISO] result ${MODEL}:"; cat "${SCRATCH}/piISO_${TAG}.json"; echo "=== DONE ${MODEL} ==="
