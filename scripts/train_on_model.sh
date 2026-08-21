#!/bin/bash
#SBATCH --job-name=xfer-train
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/opt/aar/work
#
# Train an existing AAR method (aar/ideas/<method>) on an ARBITRARY base model and
# save a merged, standalone model dir (for cross-model transfer experiments).
# Usage: sbatch --job-name=xfer-<tag> scripts/train_on_model.sh <method> <base_model> <out_dir>
set -euo pipefail
METHOD="${1:?usage: train_on_model.sh <method> <base_model> <out_dir>}"
BASE="${2:?base_model}"
OUT="${3:?out_dir}"
REPO=/opt/aar/work
export AAR_IDEAS_DIR="${AAR_IDEAS_DIR:-${TEAM_DIR:+${TEAM_DIR}/methods}}"
export PYTHONPATH="${REPO}${AAR_IDEAS_DIR:+:${AAR_IDEAS_DIR}}"
export HF_HOME=/opt/aar/work
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Gated bases (meta-llama) need a token; ungated (unsloth) ignore it harmlessly.
export HF_TOKEN="$(grep '^HF_TOKEN=' /opt/aar/work 2>/dev/null | cut -d= -f2-)"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
PY=/opt/aar/work
cd "${REPO}"
echo "[xfer] train method=${METHOD} base=${BASE} -> ${OUT}"
PYTHONUNBUFFERED=1 ${PY} -u - "${METHOD}" "${BASE}" "${OUT}" <<'PY'
import importlib, sys
method, base, out = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    mod = importlib.import_module(f"{method}.run")          # team method (AAR_IDEAS_DIR on path)
except ModuleNotFoundError:
    mod = importlib.import_module(f"aar.ideas.{method}.run")  # repo seed library
cfg = mod.MethodConfig(base_model=base, output_dir=out)
res = mod.run_experiment(cfg)
print("[xfer] DONE", res, flush=True)
PY
echo "=== DONE ${OUT} ==="
