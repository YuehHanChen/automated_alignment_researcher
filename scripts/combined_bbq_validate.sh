#!/bin/bash
#SBATCH --job-name=combval
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/aar/eval-user/combval_%j.out
R=/opt/aar/aar_harness
PY=/opt/aar/work/git/python
export PYTHONPATH="${R}"
export HF_HOME="${HF_HOME:-/opt/aar/eval-user/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/aar/eval-user/hf_datasets}"
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_GPUS=auto
set -euo pipefail
echo "[combval] $(date)"
${PY} /opt/aar/eval-user/combined_bbq_validate.py
echo "[combval] DONE-WRAPPER"
