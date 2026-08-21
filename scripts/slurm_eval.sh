#!/bin/bash
#SBATCH --job-name=aareval
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/opt/aar/work
#
# Eval job for the multi-benchmark AAR harness (fs transport).
# Reads the SECRET suite from HOLDOUT_DIR + the submitted model from
# SUBMISSIONS_DIR, scores it, writes scores.json back. Args: <run_id> <suite>.
#
# ISOLATION (production): HOLDOUT_DIR must be owned by a SEPARATE eval user,
# mode 700, and this job must run as / be able to read that user (see
# ISOLATION.md). In dev (single user) it's the honor system — the AAR is told
# not to read it, but nothing enforces that yet.

set -euo pipefail
RUN_ID="${1:?usage: slurm_eval.sh <run_id> <suite>}"
SUITE="${2:?usage: slurm_eval.sh <run_id> <suite>}"

REPO="${HARNESS_REPO:-/opt/aar/work"
export HF_HOME=/opt/aar/work
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HARNESS_TRANSPORT=fs
export PYTHONPATH="${REPO}"
# OAI_API for judge benchmarks — extract just that key (don't source the whole
# .env; the SSH-key line has spaces and breaks `source`). Kept off the research side.
ENV_FILE="${HARNESS_ENV:-/opt/aar/work"
[ -f "${ENV_FILE}" ] && export OAI_API="$(grep '^OAI_API=' "${ENV_FILE}" | cut -d= -f2-)"

cd "${REPO}"
# aar_harness has no venv of its own — reuse the safety-aar venv (torch/transformers/peft/datasets).
PY="${HARNESS_PY:-/opt/aar/work"
PYTHONUNBUFFERED=1 "${PY}" -u -m aar.eval_pod.entrypoint --run-id "${RUN_ID}" --suite "${SUITE}"
echo "=== DONE ==="
