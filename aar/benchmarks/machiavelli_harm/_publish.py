"""Publisher for machiavelli_harm — the direct-harm/violence facet (POWER-SEEKING #11). Reuses
machiavelli_power's publisher with facet="harm" (killing/physical/non_physical/intending-harm) on the train
game-split (disjoint from machiavelli_heldout, shared with machiavelli_power)."""
from __future__ import annotations

import argparse
import json

from aar.benchmarks.machiavelli_power._publish import publish_machiavelli_power


def publish_machiavelli_harm(out_path: str, n: int = 300, seed: int = 42,
                             local: str | None = None, min_gap: int = 2) -> dict:
    return publish_machiavelli_power(out_path, n=n, seed=seed, local=local, min_gap=min_gap,
                                     game_split="train", facet="harm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local", default=None)
    ap.add_argument("--min_gap", type=int, default=2)
    a = ap.parse_args()
    print(json.dumps(publish_machiavelli_harm(a.out, a.n, a.seed, a.local, a.min_gap), indent=1))
