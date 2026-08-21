#!/bin/bash
#SBATCH --job-name=judgeaudit
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/aar/eval-user/judgeaudit_%j.out
R=/opt/aar/aar_harness
PY=/opt/aar/work/git/python
export PYTHONPATH="${R}"
export HF_HOME="${HF_HOME:-/opt/aar/eval-user/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/aar/eval-user/hf_datasets}"
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
export BENCHMARK_DOCS_DIR=/opt/aar/eval-user/benchmark_docs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_GPUS=auto
for _ak in ANTHROPIC_API_KEY ANT_high_prio_API ANT_API_KEY; do
  _av="$(grep -m1 "^${_ak}=" /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
  [ -n "${_av}" ] && export "${_ak}=${_av}"
done
export JUDGE_BACKEND=anthropic JUDGE_MODEL=claude-haiku-4-5 JUDGE_CONCURRENCY=100 ANTHROPIC_MIN_INTERVAL_S=0
set -euo pipefail
echo "[judgeaudit] $(date) key=$([ -n "${ANT_high_prio_API:-}${ANTHROPIC_API_KEY:-}" ] && echo set || echo MISSING)"
${PY} /opt/aar/eval-user/judge_audit_bbqgen.py
echo "[judgeaudit] DONE-WRAPPER"
