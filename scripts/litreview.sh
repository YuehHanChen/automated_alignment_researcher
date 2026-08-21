#!/bin/bash
#SBATCH --job-name=aar-litreview
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/opt/aar/work
#
# Literature-review pre-phase: populate the team's SHARED lit forum with >=MIN
# method/paper entries before the AARs start. CPU-only — uses the Claude API +
# WebSearch/WebFetch, no GPU. Usage: sbatch scripts/litreview.sh <suite> <team_id> [min]
set -euo pipefail
SUITE="${1:?usage: litreview.sh <suite> <team_id> [min]}"
TEAM_ID="${2:?team_id}"
MIN="${3:-30}"
REPO=/opt/aar/work
export PYTHONPATH="${REPO}"
# The survey populates the AXIS-WISE literature baseline (one per safety axis), shared
# read-only by every team on that axis. Teams add their OWN in-run entries to their
# per-team LIT_FORUM_DIR (under TEAM_DIR/litreview); the survey writes the axis baseline.
# write_lit_entry targets LIT_FORUM_DIR, so point it at the axis dir for this survey job.
export LIT_FORUM_DIR="${LIT_AXIS_DIR:-/opt/aar/work"
mkdir -p "${LIT_FORUM_DIR}"
export LITREVIEW_WORKSPACE=/opt/aar/work
mkdir -p "${LITREVIEW_WORKSPACE}"
ENV=/opt/aar/work
export ANTHROPIC_API_KEY="$(grep '^ANT_high_prio_API=' "${ENV}" | cut -d= -f2-)"
export LITREVIEW_MODEL="${LITREVIEW_MODEL:-claude-sonnet-4-6}"
PY=/opt/aar/work
cd "${REPO}"
echo "[litreview] axis=${SUITE} team=${TEAM_ID} model=${LITREVIEW_MODEL} -> ${LIT_FORUM_DIR} (min ${MIN})"
PYTHONUNBUFFERED=1 ${PY} -u -m aar.litreview.run_litreview --suite "${SUITE}" --min-entries "${MIN}"
echo "[litreview] entries written: $(ls ${LIT_FORUM_DIR}/*.json 2>/dev/null | wc -l)"
echo "=== DONE ==="
