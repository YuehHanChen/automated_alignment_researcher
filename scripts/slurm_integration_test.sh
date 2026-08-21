#!/bin/bash
#SBATCH --job-name=harness-integ
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/aar/work
#
# Full submit-model loop integration test — HARDENED two-user flow. Run this as
# the RESEARCH user. It stages the BASE target model as a "submission" via
# transport.put_model, then POLLS for the composite that a separate eval worker
# (running as the secret-holding user) writes after scoring it against the
# mode-700 holdout. Since the submission IS the baseline model, the composite
# should come back ~0 headline + passes_filter — proving the transport +
# worker + entrypoint + scoring + composite path works end to end, WITHOUT the
# research side ever reading the holdout.
#
# PREREQ: an eval worker must be running. As the eval user, once:
#   sbatch scripts/eval_worker.sh sycophancy

set -euo pipefail
REPO=/opt/aar/work
export PYTHONPATH="${REPO}"
export HF_HOME=/opt/aar/work
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HARNESS_TRANSPORT=fs
# NOTE: deliberately do NOT set/read HOLDOUT_DIR or OAI_API — the research side
# can't read the holdout and doesn't run the judge; the eval worker does both.
export SUBMISSIONS_DIR=/opt/aar/work
export SCORES_DIR=/opt/aar/work
PY=/opt/aar/work
RUN_ID="integ-$(date +%s)"
cd "${REPO}"

echo "[integ] staging base model as submission ${RUN_ID}"
PYTHONUNBUFFERED=1 ${PY} -u - <<PY
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from aar import transport
m = "Qwen/Qwen2.5-3B-Instruct"
staging = tempfile.mkdtemp(prefix="submit_")
AutoTokenizer.from_pretrained(m).save_pretrained(staging)
AutoModelForCausalLM.from_pretrained(m).save_pretrained(staging)
ref = transport.put_model(staging, "${RUN_ID}")
print("[integ] put_model ->", ref, "(eval worker will pick this up)")
PY

echo "[integ] polling for the eval worker's composite (it reads the locked holdout)"
for i in $(seq 1 180); do   # up to ~1h
  if [ -f "${SCORES_DIR}/${RUN_ID}.json" ]; then
    PYTHONUNBUFFERED=1 ${PY} -u - <<PY
from aar import transport
import json
s = transport.read_scores("${RUN_ID}")
print("[integ] COMPOSITE:", json.dumps({k: s.get(k) for k in ("headline_pct","passes_filter","closed_pct")}, indent=2))
print("[integ] PASS — two-user submit-model loop returned a composite" if s and "headline_pct" in s else "[integ] FAIL")
PY
    echo "=== DONE ==="; exit 0
  fi
  sleep 20
done
echo "[integ] TIMEOUT waiting for scores — is the eval worker running? (sbatch scripts/eval_worker.sh sycophancy as the eval user)"; exit 1
