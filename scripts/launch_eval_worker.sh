#!/bin/bash
# Launch the eval worker with GPUs == the suite's benchmark count, so every
# benchmark scores in parallel (one model replica per GPU) and each iteration's
# eval finishes in ~its slowest benchmark instead of the sum. Run as the EVAL
# user (it reads the mode-700 holdout to count benchmarks).
#
# PREREQ — publish the holdout for the axis+model FIRST (writes the suite YAML this reads):
#   AXIS=sycophancy MODEL=mistral scripts/publish_holdout.sh
# Then launch the worker for the SAME axis (the model is baked into the published holdout, so
# the worker itself is model-independent — it just scores submitted models against that suite):
#   AXIS=sycophancy scripts/launch_eval_worker.sh
#
# Benchmark count = (2-3 safety) + (3 capability) = 5 or 6, so this requests
# gpu:5 or gpu:6 automatically — nothing hardcoded.
set -euo pipefail
REPO=/opt/aar/aar_repo
# Default the suite from the SAME single source the research side uses (scripts/axis_env.sh),
# so the eval worker scores the axis the chains optimize. Pass an explicit arg (or AXIS=<name>)
# to override. The held-out is read from the published suite YAML, so it stays in sync.
source "${REPO}/scripts/axis_env.sh"
SUITE="${1:-${SUITE_NAME}}"
IDLE="${2:-8000}"
HOLDOUT="${HOLDOUT_DIR:-/opt/aar/eval-user/holdout}"
YAML="${HOLDOUT}/${SUITE}/${SUITE}.yaml"
N=$(grep -c '^- name:' "${YAML}" 2>/dev/null || echo 0)
[ "${N}" -ge 1 ] || { echo "ERROR: no benchmarks in ${YAML} — publish the holdout first:" \
  "AXIS=${AXIS:-sycophancy} MODEL=${MODEL:-qwen} scripts/publish_holdout.sh" >&2; exit 1; }
echo "[launch] suite '${SUITE}' has ${N} benchmarks -> sbatch --gres=gpu:${N}"
sbatch --gres="gpu:${N}" --job-name="aar-eval-${SUITE}" "${REPO}/scripts/eval_worker.sh" "${SUITE}" "${IDLE}"
