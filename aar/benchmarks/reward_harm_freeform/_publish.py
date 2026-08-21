"""Publisher for reward_harm_freeform — the FREE-FORM agentic reward-vs-harm SCORED leg (REWARD-HACKING #9).

Renders the procedural multi-domain reward-vs-harm scenarios (`world_affecting_reward._procgen`, 19 domains)
into open-ended generation items. Seed is OFFSET from the MCQ leg's so the two ethical legs draw different
surface fills (orgs/amounts/cities) of the same domain templates — same disposition, different items.

DISJOINT from the held-outs: these are PROCEDURAL corporate/agentic dilemmas — not the keing1 source slice the
free-form held-out (`world_affecting_freeform`) holds out, and not the MACHIAVELLI text-games the structured
held-out (`machiavelli_reward`) uses. So this scored leg leaks neither held-out.

Published item schema (one per line):
  {"messages":[{"role":"user","content": <agentic dilemma + "what do you do?">]}], "hack": <unethical action>,
   "honor": <ethical action>, "domain": <scenario family>}

Usage: python -m aar.benchmarks.reward_harm_freeform._publish <out.jsonl> [--n 60] [--seed 43]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aar.benchmarks.world_affecting_reward._procgen import gen_scenarios, to_freeform


def publish(out_path: str, n: int = 60, seed: int = 43) -> dict:
    scs = gen_scenarios(n, seed)
    rows = []
    for sc in scs:
        ff = to_freeform(sc)
        rows.append({
            "messages": [{"role": "user", "content": ff["prompt"]}],
            "hack": ff["hack"],
            "honor": ff["honor"],
            "domain": ff["domain"],
        })
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    from collections import Counter
    return {"n": len(rows), "domains": len(set(r["domain"] for r in rows)),
            "per_domain": dict(Counter(r["domain"] for r in rows))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=43)
    a = ap.parse_args()
    print(json.dumps(publish(a.out, a.n, a.seed), indent=1))
