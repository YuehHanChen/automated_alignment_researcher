#!/bin/bash
# Purge ALL held-out benchmark data/scores from the RESEARCH user's space, so the AAR (research
# user, shell access) can never read the generalization-check benchmarks. Baselines necessarily
# publish the held-out to research scratch (_*baseline/) to MEASURE it; this removes it afterward.
#
# WHY NOT inside baseline_*.sh: those are parallel ARRAY jobs sharing one scratch — a per-task purge
# would delete the held-out data while another model is still loading it (race), and the held-out
# SCORES are needed post-run to patch benchmark_docs. So purge is a separate step: run after a
# baseline sweep (once held-out baselines are recorded eval-side), and launch_team.sh runs it +
# fail-safe-aborts before every team, so the AAR never starts with held-out present.
#
# Idempotent. Run as the research user.  Usage: scripts/purge_heldout_research.sh
set -uo pipefail
R=/opt/aar/work
REPO=/opt/aar/work
PY=/opt/aar/work

# Held-out benchmark names from the SINGLE SOURCE OF TRUTH (_HELD_OUT in publish_suite); fall back to
# the known set if the import is unavailable (kept in sync with scripts/publish_suite.py:_HELD_OUT).
HO=$(cd "${REPO}" 2>/dev/null && PYTHONPATH="${REPO}" "${PY}" -c "
from scripts.publish_suite import _HELD_OUT
xs=[]
for v in _HELD_OUT.values(): xs += (v if isinstance(v, list) else [v])
print(' '.join(sorted(set(xs))))" 2>/dev/null)
[ -z "${HO}" ] && HO="sycon_fp strongreject mask_factual news_factor expert_factor open_prompt_injection"
# SUPERSET files: data files that CONTAIN held-out items but aren't named the held-out bench.
# `mask.jsonl` (the full MASK set from the legacy _publish_mask) contains the held-out
# `mask_factual` (known_facts) archetype, so it must be purged too. Keep this in sync if a new
# held-out bench has a superset file. (The optimized benches use their own files, so deleting the
# superset never removes optimized data.)
HO="${HO} mask"
echo "[purge] held-out benches (+ superset files): ${HO}"

n=0
for b in ${HO}; do
  while IFS= read -r f; do [ -n "$f" ] && rm -f "$f" && n=$((n+1)); done < <(find "${R}" -maxdepth 6 -name "${b}.jsonl" 2>/dev/null)
done
# held-out FULL scores
find "${R}" -maxdepth 6 -type d -name heldout_scores -exec rm -rf {} + 2>/dev/null || true
# scratch suite yamls (they tag the held-out by name)
find "${R}" -maxdepth 5 \( -path "*baseline*/*.yaml" -o -path "*_pitest*/*.yaml" -o -path "*_pibaseline*/*.yaml" \) -exec rm -f {} + 2>/dev/null || true
# Baseline-MEASUREMENT scratch composites: _*baseline*/<...>.json (and scores_dtest) hold a per-model
# baseline INCLUDING the held-out benchmark's NAME + SCORE, and are research-readable -> a shell-capable
# AAR could cat them. Remove any such scratch .json that names a held-out. (The clean PROMPT baselines
# baselines.<axis>.<model>.json EXCLUDE the held-out and are NOT under these dirs; the AAR's own proposal
# markers in .proposals/ only CITE published benches and are left untouched. Match exact held-out names,
# NOT the 'mask' superset, so a scored 'mask_generative' file is never collaterally removed.)
_HO_RE=$(printf '%s\n' ${HO} | grep -vx mask | tr '\n' '|' | sed 's/|$//')
while IFS= read -r f; do
  [ -n "$f" ] && grep -lqE "${_HO_RE}" "$f" 2>/dev/null && rm -f "$f" && n=$((n+1))
done < <(find "${R}" -maxdepth 3 \( -path "*/_*baseline*/*.json" -o -path "*/scores_dtest/*.json" \) 2>/dev/null)
echo "[purge] removed ${n} held-out data/scratch file(s) + all heldout_scores dirs + scratch suite yamls from research"

# VERIFY (so callers can trust the exit code instead of duplicating the list -> no drift):
# re-scan for any held-out/superset data file; nonzero exit if any survive (e.g. a perms failure).
left=""
for b in ${HO}; do
  m=$(find "${R}" -maxdepth 6 -name "${b}.jsonl" 2>/dev/null | head -1)
  [ -n "${m}" ] && left="${left} ${m}"
done
# also re-scan scratch .json composites for any surviving held-out name+score leak
for b in $(printf '%s\n' ${HO} | grep -vx mask); do
  while IFS= read -r f; do
    grep -lq "${b}" "$f" 2>/dev/null && { left="${left} ${f}"; break; }
  done < <(find "${R}" -maxdepth 3 \( -path "*/_*baseline*/*.json" -o -path "*/scores_dtest/*.json" \) 2>/dev/null)
done
if [ -n "${left}" ]; then
  echo "[purge] FATAL: held-out data still present:${left}" >&2
  exit 1
fi
echo "[purge] verified: no held-out data remains on the research side"
exit 0
