#!/bin/bash
# Restart a team's RESEARCH chains CLEANLY, reusing the SAME TEAM_ID.
#
# WHY THIS EXISTS: a naive "soft restart" (scancel + resubmit the same team) leaves the
# team's PRIOR-RUN results in place — forum findings, eval scores, and the generated
# method packages under methods/. The relaunched chains then read those as "what my
# teammates already tried" (e.g. "the alpha chain scored 64.1%", "teammates tried Circuit
# Breaker RR / SimPO") even though NO chain in the new run has done anything yet. That is
# a prior-results leak. A restart must continue the team IDENTITY (same per-model holdout,
# same dashboard + eval routing, same TEAM_ID) but NOT its results.
#
# So this script, before relaunching:
#   - cancels the team's chains,
#   - ARCHIVE-PURGES the team's experimental results (forum/scores/methods/submissions/_train),
#   - resets each chain's private AGENT_LOG,
# guaranteeing the relaunched chains start with NO prior attempts visible. Literature is
# KEPT (the per-axis baseline is shared+read-only; the team's litreview is references, not a
# result). Everything purged is ARCHIVED (not deleted) under aar_archive/restart-<team>-<ts>/.
#
# Run AS THE RESEARCH user. The eval worker runs as the EVAL user — relaunch it with the
# command this script prints at the end (it serves the same TEAM_DIR queue).
#
#   TEAM_ID=refusal-qwen-20260606-221418 scripts/restart_team.sh
#   SEEDS="alpha beta gamma delta epsilon" MAX_ITERS=100 MAX_H=47 TEAM_ID=... scripts/restart_team.sh
#
# (For a fresh, brand-new team instead, use scripts/launch_team.sh — a new TEAM_ID gets an
#  empty TEAM_DIR and is clean by construction; this script is only for reusing an EXISTING id.)
set -euo pipefail
: "${TEAM_ID:?set TEAM_ID=<axis>-<model>-<timestamp>}"
REPO=/opt/aar/work
TEAM_DIR=/opt/aar/work
[ -d "${TEAM_DIR}" ] || { echo "[restart] FATAL: no such team dir ${TEAM_DIR}" >&2; exit 1; }
# axis + model are encoded in the team id: <axis>-<model>[-<tag>]-<YYYYMMDD-HHMMSS>[-<n>].
# Strip the trailing timestamp, then AXIS = first hyphen-token, MODEL = SECOND hyphen-token.
# Any extra tag between the model and the timestamp (e.g. an AAR-model tag like 'opus48') is
# IGNORED: the old `MODEL="${_base#*-}"` took EVERYTHING after the first hyphen and so yielded
# MODEL='qwen-opus48' for 'prompt_injection-qwen-opus48-<ts>', which then fails model resolution.
_base="$(printf '%s' "${TEAM_ID}" | sed -E 's/-[0-9]{8}-[0-9]{6}(-[0-9]+)?$//')"
AXIS="${_base%%-*}"; _rest="${_base#*-}"; MODEL="${_rest%%-*}"
SEEDS="${SEEDS:-alpha beta gamma delta epsilon}"
MAX_ITERS="${MAX_ITERS:-100}"; MAX_H="${MAX_H:-47}"
echo "[restart] team=${TEAM_ID}  axis=${AXIS}  model=${MODEL}  seeds='${SEEDS}'"

# 1) Cancel THIS team's chains (by job name). TWO launch shapes exist: per-seed jobs named
#    aar-<axis>-<model>-<seed> (launch_team.sh / a prior restart), and a single Slurm ARRAY
#    named aar-<axis>-<model> with elements _0.._N (the opus48 batch). Cancel BOTH forms — if
#    the array name is missed, the old chains keep running and the relaunch DUPLICATES the team
#    (every chain runs twice against the same TEAM_DIR). Decoupled training jobs drain on their own.
scancel --name="aar-${AXIS}-${MODEL}" 2>/dev/null || true
for s in ${SEEDS}; do scancel --name="aar-${AXIS}-${MODEL}-${s}" 2>/dev/null || true; done
sleep 6

# 2) ARCHIVE-PURGE the team's experimental results so the restart shows NO prior attempts.
#    Keep methods/__init__.py if present (package marker), and keep litreview/ entirely.
#    KEEP_FORUM=1 => ALSO keep forum/ (the team's SHARED, VALIDATED findings = its legitimate
#    collaborative leaderboard). This is SAFE: the prior-results leak was only ever via the raw
#    scores/ + methods/ artifacts the agents read directly (still purged here) — never the forum,
#    which holds only properly-shared findings. Use KEEP_FORUM=1 when the findings are good and you
#    just want to apply a prompt/code fix WITHOUT throwing away the research leaderboard.
ARCH="/opt/aar/work +%Y%m%d-%H%M%S)"
mkdir -p "${ARCH}"
_PURGE="scores submissions _train methods"
[ "${KEEP_FORUM:-0}" = "1" ] || _PURGE="forum ${_PURGE}"
for sub in ${_PURGE}; do
  d="${TEAM_DIR}/${sub}"
  [ -d "${d}" ] || continue
  if [ "${sub}" = "methods" ]; then
    # archive every method package but preserve a root __init__.py marker if one exists
    find "${d}" -mindepth 1 -maxdepth 1 ! -name '__init__.py' \
         -exec mv {} "${ARCH}/" \; 2>/dev/null || true
  elif [ -n "$(ls -A "${d}" 2>/dev/null)" ]; then
    mkdir -p "${ARCH}/${sub}"; mv "${d}"/* "${ARCH}/${sub}/" 2>/dev/null || true
  fi
  mkdir -p "${d}"
done
rm -f "${TEAM_DIR}"/logs/AGENT_LOG_*.md 2>/dev/null || true
echo "[restart] purged ${_PURGE} + AGENT_LOG (archived -> ${ARCH}); litreview kept$([ "${KEEP_FORUM:-0}" = "1" ] && echo '; forum/findings KEPT (KEEP_FORUM=1)')"

# 3) Relaunch the chains against the SAME TEAM_ID with freshly-empty result dirs.
export AXIS MODEL TEAM_DIR
export AAR_NO_LOCAL_GPU=true
export LIT_AXIS_DIR="/opt/aar/work"
export LOCAL_FINDINGS_DIR="${TEAM_DIR}/forum" SUBMISSIONS_DIR="${TEAM_DIR}/submissions" \
       SCORES_DIR="${TEAM_DIR}/scores" AAR_IDEAS_DIR="${TEAM_DIR}/methods" \
       SESSION_LOGS_DIR="${TEAM_DIR}/logs" LIT_FORUM_DIR="${TEAM_DIR}/litreview"
cd "${REPO}"
for s in ${SEEDS}; do
  jid=$(sbatch --parsable --gres=gpu:0 --job-name="aar-${AXIS}-${MODEL}-${s}" \
        -o "${TEAM_DIR}/logs/%x_%j.out" \
        scripts/slurm_aar_chain.sh "explore-${s}" "${MAX_H}" "${MAX_ITERS}" "${TEAM_ID}")
  echo "[restart] chain ${s} -> ${jid}"
  sleep 1
done
echo "[restart] done. Relaunch the eval worker AS THE EVAL user:"
echo "  SUBMISSIONS_DIR=${TEAM_DIR}/submissions SCORES_DIR=${TEAM_DIR}/scores \\"
echo "    sbatch --gres=gpu:3 --job-name=aar-eval-${AXIS}-${MODEL} ${REPO}/scripts/eval_worker.sh ${AXIS} 8000"
echo "[restart] (dashboard, if not already running: sbatch ${REPO}/scripts/dashboard.sh <port> ${TEAM_ID})"
