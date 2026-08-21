#!/bin/bash
#SBATCH --job-name=aar-dashboard
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --output=/opt/aar/work
#
# Live run dashboard (CPU-only, no GPU). Binds 0.0.0.0:<port> on a compute node;
# tunnel from your laptop:  ssh -L <port>:<node>:<port> <login>  then open localhost:<port>.
# Reads only research-readable data (forum/scores/logs) — never the holdout.
set -euo pipefail
PORT="${1:-8765}"
# Pin the dashboard to ONE team: pass the TEAM_ID (aar_teams/<TEAM_ID>/). Everything
# the dashboard reads — findings, scores, chain + session logs, the team's lit — is
# then scoped to that team's folder. Usage: sbatch scripts/dashboard.sh <port> <team_id>
TEAM_ID="${2:-}"
REPO=/opt/aar/work
# Dedicated dashboard venv: matplotlib + cmcrameri (real batlow) for Bruce-style
# figure rendering, + httpx/requests for the aar import chain. Kept separate from
# the training venv so the dashboard never perturbs the running team's deps.
PY=/opt/aar/work
[ -x "${PY}" ] || PY=/opt/aar/work   # fallback
if [ -n "${TEAM_ID}" ]; then
  export TEAM_DIR=/opt/aar/work
  export AAR_TEAM_ID="${TEAM_ID}"   # the dashboard reads this as the authoritative team id
                                    # (forum dir is now TEAM_DIR/forum, whose .name is just 'forum')
  # axis/model come from the TEAM_ID (<axis>-<model>-<ts>) — source axis_env.sh to
  # resolve SUITE_NAME/TARGET_MODEL/BASELINES_PATH for the raw-score baselines panel.
  export AXIS="$(printf '%s' "${TEAM_ID}" | cut -d- -f1)"
  export MODEL="$(printf '%s' "${TEAM_ID}" | cut -d- -f2)"
  source "${REPO}/scripts/axis_env.sh" 2>/dev/null || true
  # Pin every read path to THIS team's folder.
  export LOCAL_FINDINGS_DIR="${TEAM_DIR}/forum"     # findings (also drives the chain-job prefix)
  export SCORES_DIR="${TEAM_DIR}/scores"
  export CHAIN_LOGS_DIR="${TEAM_DIR}/logs"
  export SESSION_LOGS_DIR="${TEAM_DIR}/logs"
  export LIT_FORUM_DIR="${TEAM_DIR}/litreview"
  export LIT_AXIS_DIR="/opt/aar/work"
  echo "[dashboard] PINNED to team ${TEAM_ID}  (TEAM_DIR=${TEAM_DIR})"
else
  export FORUM_ROOT="/opt/aar/work"   # legacy: newest team under the old layout
  echo "[dashboard] no team_id given — legacy FORUM_ROOT mode"
fi
echo "[dashboard] node=$(hostname) port=${PORT}"
echo "[dashboard] tunnel:  ssh -L ${PORT}:$(hostname):${PORT} <login>   then open http://localhost:${PORT}"
PYTHONPATH="${REPO}" exec ${PY} -m aar.web_ui.live_dashboard --host 0.0.0.0 --port "${PORT}"
