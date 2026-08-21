#!/usr/bin/env bash
# End-to-end demo of the generic template — NO GPU, NO API keys (stub model + stub judge).
# Shows the composite (hill-climbing geomean), the held-out result (eval-private), and the
# capability gate. `stub:perfect` stands in for an improved model, `stub:weak` for a weak baseline.
set -euo pipefail
cd "$(dirname "$0")/.."            # repo root
export PYTHONPATH="$PWD"

echo "==================== stub:perfect (an 'improved' model) ===================="
python generic_aar/eval.py --suite generic_aar/suite.yaml --model stub:perfect \
    --out /tmp/generic_perfect.json --heldout-dir generic_aar/_heldout

echo ""
echo "==================== stub:weak (a weak baseline) ===================="
python generic_aar/eval.py --suite generic_aar/suite.yaml --model stub:weak \
    --out /tmp/generic_weak.json --heldout-dir generic_aar/_heldout

echo ""
echo "research-readable score (held-out stripped): /tmp/generic_perfect.json"
echo "full result incl. held-out (eval-private):    generic_aar/_heldout/generic_perfect.json"
