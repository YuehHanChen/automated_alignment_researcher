"""Publisher for rh_rubric_tamper — the AGENTIC reward-tampering SCORED leg (REWARD-HACKING #9).

The environment (Denison et al. nudged_rubric_modification) is baked into benchmark.py, so there is no
external data to vendor — the publisher just writes N episode seeds. Variation across episodes comes from
temperature-1 sampling of the same canonical RL-sandbox scenario (as in Denison's own many-trial evaluation).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def publish(out_path: str, n: int = 30, seed: int = 42) -> dict:
    rows = [{"i": i} for i in range(n)]
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"n": len(rows)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(publish(a.out, a.n, a.seed)))
