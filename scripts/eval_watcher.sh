#!/bin/bash
#SBATCH --job-name=aar-eval-watcher
#SBATCH --partition=general,overflow
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/opt/aar/eval-user/eval_watcher_%j.out
#
# EPHEMERAL EVAL watcher — SUBMIT THIS AS eval-user. Holds NO GPU (gpu:0, so it
# costs nothing against the GPU cap and nothing to preempt). It watches the submission
# queue and, for each new submission, sbatches ONE short eval JOB (eval_job.sh, gpu:2 on
# --qos=high32) that scores it and exits. So: idle => NO eval GPUs held (only this CPU
# watcher); many AARs submit at once => their eval_job's run IN PARALLEL (32-GPU high32 cap
# / 2 = up to 16 concurrent), the rest queue and start as GPUs free. Replaces the persistent
# gpu:N daemon (eval_worker.sh, kept as the fallback) which held its GPUs the whole window.
#
# Preemption-safe: if an eval job dies (preempted/failed) without producing a score, the
# watcher releases the claim and resubmits that one run-id. Multiple watchers can run
# (atomic mkdir-claim), e.g. one per per-model queue.
#
#   <suite>  : the axis (default refusal)
#   EVAL_GPUS_PER_JOB : GPUs each eval job requests (default 2)  |  EVAL_QOS (default high32)
#   SUBMISSIONS_DIR/SCORES_DIR/HOLDOUT_DIR : per-model dirs (propagated to the eval jobs)
set -uo pipefail
SUITE="${1:-refusal}"
REPO=/opt/aar/aar_repo
export HARNESS_TRANSPORT=fs
export BENCHMARK_DOCS_DIR="${BENCHMARK_DOCS_DIR:-/opt/aar/eval-user/benchmark_docs}"
export SUBMISSIONS_DIR="${SUBMISSIONS_DIR:-/opt/aar/work/aar_repo_runs/submissions}"
export SCORES_DIR="${SCORES_DIR:-/opt/aar/work/aar_repo_runs/scores}"
export HOLDOUT_DIR="${HOLDOUT_DIR-}"
EVAL_GPUS_PER_JOB="${EVAL_GPUS_PER_JOB:-2}"   # 2 GPUs/eval -> up to 16 concurrent under high32 (32-GPU cap)
EVAL_QOS="${EVAL_QOS:-high32}"                # eval-user's 32-GPU NON-preemptible QOS (the blocking AAR
                                             # poll won't starve); set EVAL_QOS=low for the old uncapped/preemptible mode
# Per-team org (mirror of eval_worker.sh): derive the per-MODEL HOLDOUT + eval-PRIVATE held-out scores from
# SUBMISSIONS_DIR, and EXPORT them so each sbatch'd eval_job.sh inherits the right holdout via default
# propagation (a same-axis/different-model team reads ITS OWN holdout). Self-sufficient — no launcher env needed.
case "${SUBMISSIONS_DIR}" in
  */aar_teams/*/submissions)
    EVAL_TEAM_ID="$(basename "$(dirname "${SUBMISSIONS_DIR}")")"
    EVAL_TEAMS="/opt/aar/work"
    EVAL_TEAM_DIR="${EVAL_TEAMS}/${EVAL_TEAM_ID}"
    mkdir -p "${EVAL_TEAM_DIR}/heldout_scores" "${EVAL_TEAM_DIR}/_evalwork"
    chmod 700 "${EVAL_TEAMS}" "${EVAL_TEAM_DIR}" "${EVAL_TEAM_DIR}/heldout_scores" "${EVAL_TEAM_DIR}/_evalwork" 2>/dev/null || true
    export HELDOUT_SCORES_DIR="${EVAL_TEAM_DIR}/heldout_scores"
    export HARNESS_RUNS_DIR="${EVAL_TEAM_DIR}"
    _AXIS="${EVAL_TEAM_ID%%-*}"
    _MTAG="$(printf '%s' "${EVAL_TEAM_ID#${_AXIS}-}" | sed -E 's/-[0-9]{8}-[0-9]{6}(-[0-9]+)?$//')"
    export HOLDOUT_DIR="${HOLDOUT_DIR:-/opt/aar/work"
    echo "[watcher] team=${EVAL_TEAM_ID} HOLDOUT_DIR=${HOLDOUT_DIR} (model_tag=${_MTAG}, axis=${_AXIS})"
    ;;
esac
export HOLDOUT_DIR="${HOLDOUT_DIR:-/opt/aar/eval-user/holdout}"
export HELDOUT_SCORES_DIR="${HELDOUT_SCORES_DIR:-${HOLDOUT_DIR}/heldout_scores}"
CLAIMS="${SCORES_DIR}/.claims"          # one claim dir per rid (shared with eval_worker.sh's scheme)
JOBS="${SCORES_DIR}/.evaljobs"          # rid -> sbatch job id, to detect a died-without-score job
mkdir -p "${CLAIMS}" "${JOBS}" 2>/dev/null || true
echo "[watcher $(hostname)] queue=${SUBMISSIONS_DIR} -> spawns gpu:${EVAL_GPUS_PER_JOB} ${EVAL_QOS}-QoS eval jobs (suite ${SUITE})"

while true; do
  # Re-create the claim/jobs dirs EACH loop — robust if they're cleared underneath us
  # (mkdir of a claim fails silently without the parent, which would stop all spawns).
  mkdir -p "${CLAIMS}" "${JOBS}" 2>/dev/null || true
  for d in "${SUBMISSIONS_DIR}"/*/; do
    [ -d "$d" ] || continue
    rid=$(basename "$d")
    case "$rid" in _*|.*) continue;; esac
    [ -f "${d}.submitted" ] || continue
    [ -f "${SCORES_DIR}/${rid}.json" ] && continue          # already scored
    if mkdir "${CLAIMS}/${rid}" 2>/dev/null; then
      # fresh -> spawn one eval job (HOLDOUT/SUBMISSIONS/SCORES propagate from this env)
      jid=$(sbatch --parsable --qos="${EVAL_QOS}" --gres="gpu:${EVAL_GPUS_PER_JOB}" \
                   --job-name="aar-evaljob-${SUITE}" \
                   "${REPO}/scripts/eval_job.sh" "${rid}" "${SUITE}" 2>/dev/null)
      echo "${jid}" > "${JOBS}/${rid}" 2>/dev/null || true
      echo "[watcher] ${rid} -> eval job ${jid:-FAILED} (gpu:${EVAL_GPUS_PER_JOB}, ${EVAL_QOS})"
    else
      # claimed already: if its eval job is gone but no score exists, it died -> resubmit
      jf="${JOBS}/${rid}"
      [ -f "$jf" ] || continue
      [ -f "${SCORES_DIR}/${rid}.json" ] && continue
      jid=$(cat "$jf" 2>/dev/null)
      [ -n "$jid" ] || continue
      if ! squeue -j "$jid" -h -o "%T" 2>/dev/null | grep -q .; then
        echo "[watcher] eval job ${jid} for ${rid} died w/o score -> releasing claim to resubmit"
        rm -rf "${CLAIMS}/${rid}" 2>/dev/null || true
        rm -f  "${JOBS}/${rid}"   2>/dev/null || true
      fi
    fi
  done
  sleep 15
done
