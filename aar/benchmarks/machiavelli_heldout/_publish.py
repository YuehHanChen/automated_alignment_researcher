"""Publisher for machiavelli_heldout — the game-disjoint MACHIAVELLI held-out probe (POWER-SEEKING #11).
Reuses machiavelli_power's publisher with game_split="heldout" (games whose stable bucket==0), so its games
are DISJOINT from the scored machiavelli_power set (game_split="train")."""
from __future__ import annotations

import argparse
import json

from aar.benchmarks.machiavelli_power._publish import publish_machiavelli_power


def publish_machiavelli_heldout(out_path: str, n: int = 300, seed: int = 42,
                                local: str | None = None, min_gap: int = 2) -> dict:
    # facet="all" = the BROAD gratuitous-harm probe (covers both scored facets: acquisition + direct-harm) on
    # the disjoint held-out games -> a generalization read over the whole property, not just one facet.
    return publish_machiavelli_power(out_path, n=n, seed=seed, local=local, min_gap=min_gap,
                                     game_split="heldout", facet="all")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local", default=None)
    ap.add_argument("--min_gap", type=int, default=2)
    a = ap.parse_args()
    print(json.dumps(publish_machiavelli_heldout(a.out, a.n, a.seed, a.local, a.min_gap), indent=1))
