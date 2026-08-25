#!/bin/bash
# Launch the eval DISPATCHER (watcher) — run as the EVAL user (eval-user). It holds
# ZERO GPUs and, for every submission, sbatches a one-shot gpu:2 eval_job.sh on --qos=high32,
# so evals run IN PARALLEL (up to 16 at once under the 32-GPU cap) and idle => 0 eval GPUs.
# This is the dynamic replacement for launch_eval_worker.sh (the persistent gpu:N daemon, which
# held its GPUs the whole window). Same isolation: jobs run as this eval user, read the mode-700
# holdout; the AAR only stages a model and polls SCORES_DIR.
#
# PREREQ — publish the holdout for the axis+model FIRST (writes the suite YAML the eval reads):
#   AXIS=sycophancy MODEL=mistral scripts/publish_holdout.sh
# Then (per-team: set SUBMISSIONS_DIR/SCORES_DIR to the team's dirs, same as the old worker):
#   AXIS=sycophancy scripts/launch_eval_watcher.sh   [gpus_per_eval=2]
set -euo pipefail
REPO=/opt/aar/aar_repo
source "${REPO}/scripts/axis_env.sh"      # SUITE_NAME from the single axis source (== what the chains optimize)
SUITE="${1:-${SUITE_NAME}}"
GPE="${2:-2}"
HOLDOUT="${HOLDOUT_DIR:-/opt/aar/eval-user/holdout}"
YAML="${HOLDOUT}/${SUITE}/${SUITE}.yaml"
[ -f "${YAML}" ] || { echo "ERROR: no suite at ${YAML} — publish the holdout first:" \
  "AXIS=${AXIS:-sycophancy} MODEL=${MODEL:-qwen} scripts/publish_holdout.sh" >&2; exit 1; }
echo "[launch] eval WATCHER for '${SUITE}': 0 GPU; dispatches gpu:${GPE} eval_job's on qos=${EVAL_QOS:-high32}"
# gpu:0 => the watcher costs nothing against the GPU cap. EVAL_GPUS_PER_JOB/EVAL_QOS propagate
# to the watcher (and onward to each eval_job) via sbatch's default env propagation.
EVAL_GPUS_PER_JOB="${GPE}" EVAL_QOS="${EVAL_QOS:-high32}" \
  sbatch --gres=gpu:0 --job-name="aar-evalwatch-${SUITE}" "${REPO}/scripts/eval_watcher.sh" "${SUITE}"
