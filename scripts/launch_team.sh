#!/bin/bash
# Launch ONE TEAM of AARs that share a fresh, private forum AND a fresh method
# workspace — so a team can't see previous teams' forum, methods, or AGENT_LOGs
# (true per-team independence; needed for per-seed forecasting validity).
#
#   [AXIS=<axis>] [MODEL=<model>] scripts/launch_team.sh ["seed1 seed2 ..."] [max_iters] [max_hours]
#   e.g. scripts/launch_team.sh "alpha beta gamma" 100 47                 # default: sycophancy on qwen
#        AXIS=sycophancy MODEL=mistral scripts/launch_team.sh "a b" 100 47
#        AXIS=refusal    MODEL=llama   scripts/launch_team.sh "a b"        # needs scripts/axis/refusal.env
#   AXIS  selects the safety axis (scripts/axis/<AXIS>.env; 'sycophancy' is built in).
#   MODEL selects which of the 6 models to optimize (qwen|mistral|llama|olmo|gemma|phi, or a full HF id).
#   The chains inherit the SAME axis AND model (one source: scripts/axis_env.sh + models.sh).
#
# OPTIONAL — seed this team's forum from a PRIOR team's findings (opt-in; default
# is a fresh empty forum). The prior findings are COPIED in, so the new team
# starts already aware of what's been tried (e.g. "diverse_antisyc_sft got 15.5%
# but failed the capability filter") and builds on it — while the source forum is
# left untouched and this team still accumulates into its OWN new forum.
#   SEED_FORUM_FROM=<TEAM_ID>  scripts/launch_team.sh sycophancy
#   SEED_FORUM_FROM=latest     scripts/launch_team.sh sycophancy   # most recent prior team
# List prior teams to pick from:
#   ls -1dt /opt/aar/work
#
# Run the eval worker once as the EVAL user (it serves all teams):
#   scripts/launch_eval_worker.sh <suite>
set -euo pipefail
AXIS="${AXIS:-sycophancy}"     # safety axis  -> scripts/axis/<AXIS>.env ('sycophancy' = built-in default)
MODEL="${MODEL:-qwen}"         # target model -> one of the 6 (scripts/models.sh); independent of AXIS
export AXIS MODEL
REPO=/opt/aar/work
# SINGLE SOURCE OF TRUTH for axis + model — the same file slurm_aar_chain.sh sources, so the
# team launcher and its chains can never disagree (the old positional SUITE arg did).
source "${REPO}/scripts/axis_env.sh"   # sets SUITE_NAME, HELD_OUT_BENCH, SAFETY_*, SEED_METHOD, TARGET_MODEL
SUITE="${SUITE_NAME}"
SEEDS="${1:-alpha beta gamma}"
MAX_ITERS="${2:-500}"          # per-CHAIN session cap (raised 100->500 2026-06-20); the 47h walltime is usually the binding limit per cycle
MAX_H="${3:-47}"
# Team id includes the model so different-model runs of the same axis get separate forums.
# Sanitize to a SINGLE safe path component: a MODEL passed as a full HF id ('org/name')
# would otherwise put a '/' in the id and break the path (and make the atomic mkdir below
# loop forever on a non-existent parent). tr maps anything but [A-Za-z0-9._-] to '_'.
# Agent-model tag (claude-opus-4-8 -> opus48): records WHICH research model produced this team —
# invaluable when a model is retired mid-fleet (cf. claude-fable-5). Hyphen-free so the positional
# parsers (dashboard.sh cut -f1/-f2; live_dashboard) keep axis=field1, model=field2, agent=field3.
_AGENT_TAG="$(printf '%s' "${AAR_AGENT_MODEL:-claude-opus-4-8}" | sed 's/^claude-//; s/[^A-Za-z0-9]//g')"
TEAM_ID="$(printf '%s' "${SUITE}-${MODEL}-${_AGENT_TAG}-$(date +%Y%m%d-%H%M%S)" | tr -c 'A-Za-z0-9._-' '_')"
# UNIQUENESS: the timestamp is only second-granular, so two same-axis+model launches
# in the SAME second would mint the IDENTICAL id and share one TEAM_DIR. CLAIM the id
# ATOMICALLY with `mkdir` (no -p): it fails if the dir already exists, so even two
# truly concurrent launches each get a distinct id (no check-then-create race). The
# folder is the authoritative claim. [[per-team folder layout]]
_AAR_TEAMS=/opt/aar/work
mkdir -p "${_AAR_TEAMS}"
_base="${TEAM_ID}"; _n=1
until mkdir "${_AAR_TEAMS}/${TEAM_ID}" 2>/dev/null; do
  # Bounded so a persistent mkdir failure (bad perms, full FS) fails LOUDLY instead of
  # spinning forever — it is NOT a collision in that case.
  [ "${_n}" -gt 1000 ] && { echo "[team] FATAL: cannot claim a unique TEAM_ID under ${_AAR_TEAMS} (perms? disk full?)" >&2; exit 1; }
  TEAM_ID="${_base}-${_n}"; _n=$((_n+1))
done
# ===========================================================================
# ONE FOLDER PER TEAM. Everything this team produces — findings, the train/eval
# queue, generated method code, logs, and the team's OWN in-run literature —
# lives under TEAM_DIR=aar_teams/<TEAM_ID>/. (Exception: the axis-wise literature
# BASELINE is shared per-axis at aar_litreview/<axis>/, read-only to every team.)
# ===========================================================================
export TEAM_DIR="${_AAR_TEAMS}/${TEAM_ID}"
FORUM="${TEAM_DIR}/forum"
mkdir -p /opt/aar/work "${TEAM_DIR}"/{forum,submissions,scores,methods,logs,litreview,_train}
# Cross-user channel perms: the eval user (also in group `shared`) must TRAVERSE
# TEAM_DIR and READ submissions / WRITE scores. Mirror the legacy 2770-setgid queue
# dirs; keep the team's other subdirs group-traversable-only.
chgrp shared /opt/aar/work "${TEAM_DIR}" "${TEAM_DIR}/submissions" "${TEAM_DIR}/scores" 2>/dev/null || true
chmod 2750 /opt/aar/work "${TEAM_DIR}" 2>/dev/null || true
chmod 2770 "${TEAM_DIR}/submissions" "${TEAM_DIR}/scores" 2>/dev/null || true
# Export the per-team dirs so chains, the train jobs they spawn, and the dashboard
# all agree (config.py also derives these from TEAM_DIR — belt and suspenders).
export LOCAL_FINDINGS_DIR="${TEAM_DIR}/forum"
export SUBMISSIONS_DIR="${TEAM_DIR}/submissions"
export SCORES_DIR="${TEAM_DIR}/scores"
export AAR_IDEAS_DIR="${TEAM_DIR}/methods"
export SESSION_LOGS_DIR="${TEAM_DIR}/logs"
export LIT_FORUM_DIR="${TEAM_DIR}/litreview"                          # team's OWN in-run lit (private)
export LIT_AXIS_DIR=/opt/aar/work    # axis baseline (shared, read-only)
cd "${REPO}"
echo "[team] TEAM_ID=${TEAM_ID}"
echo "[team] axis=${SUITE}  model=${MODEL} (${TARGET_MODEL})"
echo "[team] TEAM_DIR=${TEAM_DIR}  (forum+submissions+scores+methods+logs+litreview all here)"

# --- HELD-OUT ISOLATION (enforced before ANY AAR starts). Baselines leave the held-out benchmarks'
# data/scores in research scratch (_*baseline/); the AAR runs as THIS user with shell access, so it
# could read them and game the generalization check. The purge script removes them AND verifies
# (nonzero exit if any survive) — we trust its exit code rather than re-listing the held-out names
# here (one source of truth, no drift). Refuse to launch if it can't guarantee a clean research side.
if ! bash "${REPO}/scripts/purge_heldout_research.sh"; then
  echo "[team] FATAL: held-out data still present on the research side — REFUSING to launch (the AAR could read it)." >&2
  exit 1
fi
echo "[team] held-out isolation: research side verified clean of held-out data/scores"

# --- ISOLATION: archive previous teams' generated methods + transcripts OUT of
# the live workspace so THIS team starts fresh and cannot read prior teams' work.
# Keeps the provided seed library (TEMPLATE + the axis's ${SEED_METHOD}). Archived, NOT deleted.
# SKIP_ISOLATION=1 => PARALLEL-TEAM mode: ANOTHER team is already running in this
# same workspace (e.g. a different MODEL on the same axis). Do NOT archive — that
# would move the OTHER live team's methods/logs/.out out from under it and break it.
# The two teams coexist safely because EVERY mutable name is MODEL-namespaced:
# session ids + AGENT_LOG + method-dir prefix (slurm_aar_chain.sh IDEA_NAME/CHAIN_TAG),
# the forum + lit forum (per-TEAM_ID), and submissions/scores/holdout (per-model env).
if [ -n "${TEAM_DIR:-}" ]; then
  # Per-team layout: generated methods live in ${TEAM_DIR}/methods, so no per-team
  # archiving is needed. BUT the shared aar/ideas/ is advertised to every AAR as the
  # method SCAFFOLD — it must hold ONLY the structural TEMPLATE, never leftover method
  # dirs from earlier runs (an AAR would mistake those for prescribed "seed ideas",
  # which we deliberately do NOT give — methods are derived from the lit review).
  # Sweep any stray dirs/files out to the archive. Idempotent; safe in parallel-team
  # mode because no live team's methods are ever in aar/ideas/ (they're in TEAM_DIR).
  ARCHIVE=/opt/aar/work
  mkdir -p "${ARCHIVE}/ideas"
  find "${REPO}/aar/ideas" -mindepth 1 -maxdepth 1 \
       ! -name TEMPLATE ! -name __init__.py ! -name __pycache__ \
       -exec mv {} "${ARCHIVE}/ideas/" \; 2>/dev/null || true
  echo "[team] per-team methods at ${TEAM_DIR}/methods; aar/ideas swept to scaffold-only: $(find "${REPO}/aar/ideas" -mindepth 1 -maxdepth 1 ! -name __pycache__ -printf '%f ' 2>/dev/null)"
elif [ "${SKIP_ISOLATION:-0}" = "1" ]; then
  echo "[team] SKIP_ISOLATION=1 — parallel-team mode: NOT archiving (sharing the live workspace with another running team; model-namespacing keeps them separate)"
else
  ARCHIVE=/opt/aar/work
  mkdir -p "${ARCHIVE}/ideas" "${ARCHIVE}/session_logs" "${ARCHIVE}/job_logs"
  # archive EVERYTHING in aar/ideas/ (method dirs AND stray files like a top-level
  # AGENT_LOG.md) except the provided seed library + package init. The axis's seed
  # (SEED_METHOD, from axis_env.sh) is kept dynamically — not hardcoded to one axis.
  KEEP=( -name TEMPLATE -o -name __init__.py -o -name __pycache__ )
  [ -n "${SEED_METHOD:-}" ] && KEEP+=( -o -name "${SEED_METHOD}" )
  find "${REPO}/aar/ideas" -mindepth 1 -maxdepth 1 \
       ! \( "${KEEP[@]}" \) -exec mv {} "${ARCHIVE}/ideas/" \; 2>/dev/null || true
  find "${REPO}/aar/research_loop/logs" -maxdepth 1 \
       \( -name "session_*.log" -o -name "AGENT_LOG_*.md" \) \
       -exec mv {} "${ARCHIVE}/session_logs/" \; 2>/dev/null || true
  find /opt/aar/work -maxdepth 1 -name "aar-${SUITE}-*.out" \
       -exec mv {} "${ARCHIVE}/job_logs/" \; 2>/dev/null || true
  echo "[team] isolated: archived prior methods+transcripts -> ${ARCHIVE}"
  echo "[team] aar/ideas now: $(find "${REPO}/aar/ideas" -mindepth 1 -maxdepth 1 ! -name __pycache__ -printf '%f ' 2>/dev/null)"
fi

# --- OPTIONAL: seed this team's forum from a prior team (opt-in via SEED_FORUM_FROM).
# Copies the source team's findings into THIS team's fresh forum so it starts with
# prior knowledge but accumulates independently; the source forum is never mutated.
if [ -n "${SEED_FORUM_FROM:-}" ]; then
  SRC_TEAM="${SEED_FORUM_FROM}"
  if [ "${SRC_TEAM}" = "latest" ]; then
    # newest prior team's forum (aar_teams/<team>/forum), excluding the one we just made
    SRC_TEAM=$(ls -1dt /opt/aar/work 2>/dev/null \
               | grep -v "/${TEAM_ID}/" | head -1 | sed 's#/forum/$##' | xargs -n1 basename 2>/dev/null || true)
  fi
  # findings now live under aar_teams/<team>/forum (new layout); fall back to the legacy
  # aar_forum/<team> for teams created before the consolidation.
  SRC="/opt/aar/work"
  [ -d "${SRC}" ] || SRC="/opt/aar/work"
  if [ -n "${SRC_TEAM}" ] && [ -d "${SRC}" ]; then
    n=$(find "${SRC}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    # findings (JSON) + their code snapshots (<id>_<idea>_code/ dirs)
    cp "${SRC}"/*.json "${FORUM}/" 2>/dev/null || true
    cp -r "${SRC}"/*_code "${FORUM}/" 2>/dev/null || true
    c=$(find "${FORUM}" -maxdepth 1 -type d -name '*_code' 2>/dev/null | wc -l | tr -d ' ')
    echo "[team] SEEDED forum from prior team '${SRC_TEAM}': ${n} finding(s) + ${c} code snapshot(s) copied (source untouched)"
  else
    echo "[team] WARNING: SEED_FORUM_FROM='${SEED_FORUM_FROM}' not found — starting with an EMPTY forum."
    echo "[team] available teams: $(ls -1dt /opt/aar/work 2>/dev/null | sed 's#/forum/$##' | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
  fi
fi

# --- LITERATURE-REVIEW: the AXIS-WISE baseline survey (>=MIN safety-training methods —
# general + axis-specific) lives at LIT_AXIS_DIR=aar_litreview/<axis>/, shared READ-ONLY by
# every team on this axis so they ground their designs in known work. Built once per axis.
# The team's OWN in-run additions go to TEAM_DIR/litreview (private to the team) — NOT here,
# so one team's discoveries never leak into another team's baseline.
#   LITREVIEW_REFRESH=1  re-survey + overwrite the axis baseline (after a lit schema change)
#   LITREVIEW_SKIP=1     skip the survey entirely
#   LITREVIEW_MIN=<n>    target entry count (default 30)
LIT_AXIS="${LIT_AXIS_DIR}"
mkdir -p "${LIT_AXIS}"
_lit_n=$(find "${LIT_AXIS}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
# RECOVERY: a prior run may have surveyed this axis under a legacy location (the old per-axis
# _cache/<axis>/, an old per-team aar_litreview/<axis>-*/ forum, or a prior team's
# aar_teams/<axis>-*/litreview/). Reuse it rather than re-survey — a survey is ~40 min and is
# identical per axis. [[lit-cache-vs-forum]]
if [ "${LITREVIEW_SKIP:-0}" != "1" ] && [ "${LITREVIEW_REFRESH:-0}" != "1" ] && [ "${_lit_n}" -lt "${LITREVIEW_MIN:-30}" ]; then
  for _src in /opt/aar/work \
              $(ls -1dt /opt/aar/work 2>/dev/null) \
              $(ls -1dt /opt/aar/work 2>/dev/null); do
    case "${_src%/}" in "${LIT_AXIS%/}") continue;; esac
    [ -d "${_src%/}" ] || continue   # FRESH AXIS: _cache/<axis> (first candidate) won't exist; find on a
                                     # missing path exits 1, and under `set -euo pipefail` (line 26) the
                                     # `_pn=$(... | ...)` command-sub failure kills the whole launch SILENTLY
                                     # before the survey is ever submitted. Skip non-existent candidates.
    _pn=$({ find "${_src}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' '; } || true)
    if [ "${_pn:-0}" -ge "${LITREVIEW_MIN:-30}" ]; then
      cp "${_src%/}"/*.json "${LIT_AXIS}/" 2>/dev/null || true
      _lit_n=$(find "${LIT_AXIS}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
      echo "[team] RECOVERED ${_lit_n} axis-baseline lit entries from ${_src%/} (no re-survey)"
      break
    fi
  done
fi
if [ "${LITREVIEW_SKIP:-0}" = "1" ]; then
  echo "[team] LITREVIEW_SKIP=1 — skipping the literature survey"
elif [ "${LITREVIEW_REFRESH:-0}" != "1" ] && [ "${_lit_n}" -ge "${LITREVIEW_MIN:-30}" ]; then
  echo "[team] reusing AXIS baseline literature for '${SUITE}' (${_lit_n} entries, ${LIT_AXIS}) — no re-survey (LITREVIEW_REFRESH=1 to rebuild)"
else
  [ "${LITREVIEW_REFRESH:-0}" = "1" ] && { echo "[team] LITREVIEW_REFRESH=1 — re-surveying axis '${SUITE}'"; rm -f "${LIT_AXIS}"/*.json 2>/dev/null || true; }
  echo "[team] literature survey (fresh, >=${LITREVIEW_MIN:-30} entries) — blocking until done..."
  sbatch --wait --job-name="aar-litreview-${SUITE}" \
         scripts/litreview.sh "${SUITE}" "${TEAM_ID}" "${LITREVIEW_MIN:-30}" || \
    echo "[team] WARNING: litreview job returned non-zero (continuing)"
fi
n_lit=$(find "${LIT_AXIS}" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
echo "[team] axis literature baseline ready: ${n_lit} entries at ${LIT_AXIS}  (team adds its own to ${LIT_FORUM_DIR})"

# DECOUPLED (DEFAULT = 1 — gpu:0 GPU-LESS agents): the chain holds NO GPU (--gres=gpu:0) and trains
# each method via a SEPARATE low-QoS GPU job (slurm_train_submit.sh). This is the DEFAULT and the only
# mode that scales: an agent must NEVER hold a GPU just to think/poll. Agents cost 0 GPUs, so many
# teams run in parallel without hitting the per-user GPU cap (QOSMaxGRESPerUser); training rides the
# uncapped `low` queue. Set DECOUPLED=0 ONLY for a deliberate SINGLE inline-capable team (legacy gpu:1)
# — NEVER run inline alongside other teams (it maxes the GPU cap and starves everyone's training).
if [ "${DECOUPLED:-1}" = "1" ]; then
  export AAR_NO_LOCAL_GPU=true     # -> prompt renders the "no local GPU, train via job" branch
  CHAIN_GRES_ARG="--gres=gpu:0"
  echo "[team] DECOUPLED (default) — gpu:0 GPU-less agents; per-method training via separate low-QoS jobs"
else
  CHAIN_GRES_ARG=""
  echo "[team] ⚠️ DECOUPLED=0 — INLINE gpu:1 agents (legacy): each agent HOLDS a GPU. Only safe as a SINGLE team; do NOT run alongside other teams."
fi

for s in ${SEEDS}; do
  # AXIS + MODEL are already exported at the top of this script, so Slurm's DEFAULT env
  # propagation carries them to the chain (which resolves the axis/model from axis_env.sh).
  # CRITICAL: do NOT write --export="ALL,AXIS=..,MODEL=..". That "ALL,VAR=val" form is
  # mishandled by this cluster's Slurm — it mangles the job's environment, so the chain shows
  # RUNNING but the agent hangs at init with NO output (no .out, no session log). TEAM_ID is
  # also passed positionally below, so nothing axis-specific depends on the env alone.
  jid=$(sbatch --parsable ${CHAIN_GRES_ARG} --job-name="aar-${SUITE}-${MODEL}-${s}" \
        -o "${TEAM_DIR}/logs/%x_%j.out" \
        scripts/slurm_aar_chain.sh "explore-${s}" "${MAX_H}" "${MAX_ITERS}" "${TEAM_ID}")
  echo "[team] chain ${s} -> job ${jid}"
done
echo "[team] EVAL SIDE (as eval user): publish the holdout for THIS axis+model, then run the worker"
echo "[team]   pointed at THIS team's queue. The worker derives the eval-side per-team folder from"
echo "[team]   SUBMISSIONS_DIR, so the held-out scores + eval work + log land in the eval user's own"
echo "[team]   aar_teams/${TEAM_ID}/ (mode-700). The HOLDOUT itself stays per-axis-model."
echo "[team]   AXIS=${AXIS} MODEL=${MODEL} scripts/publish_holdout.sh"
echo "[team]   SUBMISSIONS_DIR=${TEAM_DIR}/submissions SCORES_DIR=${TEAM_DIR}/scores HOLDOUT_DIR=<holdout|holdout_${MODEL}> \\"
echo "[team]     sbatch --gres=gpu:3 --job-name=aar-eval-${SUITE}-${MODEL} scripts/eval_worker.sh ${SUITE} 8000"
echo "[team] (dashboard: sbatch scripts/dashboard.sh <port> ${TEAM_ID})"
echo "[team] TEAM_ID=${TEAM_ID}"
