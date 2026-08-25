#!/bin/bash
# verify_heldout_isolation.sh — RUN AS THE RESEARCH USER (the user the AAR runs as).
#
# Asserts the held-out benchmark's SCORE is unreachable from the research/AAR side:
#   [1] the eval-private dirs (HELDOUT_SCORES_DIR, HOLDOUT_DIR) are NOT readable here;
#   [2] the held-out benchmark id does NOT appear in any research-readable score/finding;
#   [3] `held_out_pct` does NOT appear in the research-readable handoff scores.
# Exit 0 = isolated. Non-zero = a LEAK or a broken boundary (investigate immediately).
#
# Usage: scripts/verify_heldout_isolation.sh [held_out_id]   (default: sycon_fp)
# Resolve dirs from the live aar config; override the repo/venv via AAR_REPO / AAR_PY.
# NB: scans DATA dirs (scores/findings/evalwork), not source code — the plugin source
# naming the benchmark is expected and is not a SCORE leak.
set -uo pipefail
HELD="${1:-sycon_fp}"
R="${AAR_REPO:-/opt/aar/aar_repo}"
PY="${AAR_PY:-/opt/aar/work/git/python}"

eval "$(PYTHONPATH="$R" "$PY" - <<'PYEOF'
from aar import config
for k in ("SCORES_DIR", "LOCAL_FINDINGS_DIR", "HARNESS_RUNS_DIR", "HELDOUT_SCORES_DIR", "HOLDOUT_DIR"):
    print(f'{k}={getattr(config, k, "")}')
PYEOF
)"

fail=0
echo "=== held-out isolation check (id='$HELD', user=$(whoami)) ==="

echo "[1] eval-private dirs MUST be unreadable by this user:"
for d in "$HELDOUT_SCORES_DIR" "$HOLDOUT_DIR"; do
  if ls -A "$d" >/dev/null 2>&1; then
    echo "  FAIL  READABLE: $d  (kernel boundary broken — held-out/holdout exposed)"; fail=1
  else
    echo "  ok    not readable: $d"
  fi
done

echo "[2] held-out id MUST be absent from research-readable score/finding artifacts:"
for d in "$SCORES_DIR" "$LOCAL_FINDINGS_DIR" "$HARNESS_RUNS_DIR/_evalwork"; do
  [ -d "$d" ] || { echo "  --    (absent, skip): $d"; continue; }
  hits=$(grep -rIl --include="*.json" -- "$HELD" "$d" 2>/dev/null)
  if [ -n "$hits" ]; then
    echo "  FAIL  '$HELD' found in:"; echo "$hits" | sed 's/^/          /'; fail=1
  else
    echo "  ok    '$HELD' absent under: $d"
  fi
done

echo "[3] 'held_out_pct' MUST be absent from the research-readable handoff scores:"
hits=$(grep -rIl --include="*.json" "held_out_pct" "$SCORES_DIR" 2>/dev/null)
if [ -n "$hits" ]; then
  echo "  FAIL  held_out_pct leaked into the handoff:"; echo "$hits" | sed 's/^/          /'; fail=1
else
  echo "  ok    no held_out_pct in the handoff ($SCORES_DIR)"
fi

echo "==="
if [ "$fail" = 0 ]; then
  echo "PASS — the held-out is isolated from the research/AAR side."
else
  echo "FAIL — held-out score is reachable from the research side. DO NOT trust the generalization check until fixed."
fi
exit $fail
