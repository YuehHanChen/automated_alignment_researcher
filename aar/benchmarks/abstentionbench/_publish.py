"""Publisher for abstentionbench — broad-abstention, a leg of the CONCEALING-UNCERTAINTY axis (#10).

Source: **AbstentionBench** (Kirichenko et al., Meta; `facebook/AbstentionBench`). The full benchmark is 20+
constituent datasets, but its HF release ships only a **datasets LOADER SCRIPT** (rejected by `datasets`
>=4) and most constituents need local data downloads or `trust_remote_code` (also 4.x-blocked), so a faithful
FULL build is infeasible in this environment. We build a **documented SUBSET** from the three constituents
that are **natively bi-labeled** (each ships both should_abstain AND should-answer items, so per-dataset F1 is
well-defined without the source's underspecification-perturbation logic) and load directly from HF:
  - **SQuAD2** (`rajpurkar/squad_v2`): answerable vs is_impossible (unanswerable reading-comp).
  - **SelfAware** (`JesusCrist/selfaware`): answerable flag (answerable vs unanswerable/unknown).
  - **KUQ** (`amayuelas/KUQ`): known vs unknown questions.
The source's GSM8K/MMLU/GPQA "controls" are **underspecified-PERTURBED** variants (not raw should-answer) →
they need the source perturbation logic and are OMITTED here (documented). Balanced 50 should_abstain + 50
should_answer per dataset (seed 42) so per-dataset F1 has both classes.

Item schema (one per line):
  {"prompt": <question (+context for SQuAD2)>, "should_abstain": bool, "dataset": str, "id": str}
(The reference answer / gold abstention label are NOT shipped — the abstain-detector judge runs BLIND.)

Usage: python -m aar.benchmarks.abstentionbench._publish <out.jsonl> [--n 300] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _squad2(rng, k):
    from datasets import load_dataset
    d = list(load_dataset("rajpurkar/squad_v2", split="validation"))
    rng.shuffle(d)
    pos, neg = [], []                              # pos = should_abstain (unanswerable)
    for it in d:
        unanswerable = len(it["answers"]["text"]) == 0
        row = {"prompt": f"{it['context'].strip()}\nQuestion: {it['question'].strip()}",
               "should_abstain": unanswerable, "dataset": "SQuAD2"}
        (pos if unanswerable else neg).append(row)
        if len(pos) >= k and len(neg) >= k:
            break
    return pos[:k] + neg[:k]


def _selfaware(rng, k):
    from datasets import load_dataset
    d = list(load_dataset("JesusCrist/selfaware", split="train"))
    rng.shuffle(d)
    pos, neg = [], []
    for it in d:
        abst = not bool(it["answerable"])          # unanswerable -> should abstain
        row = {"prompt": str(it["question"]).strip(), "should_abstain": abst, "dataset": "SelfAware"}
        (pos if abst else neg).append(row)
        if len(pos) >= k and len(neg) >= k:
            break
    return pos[:k] + neg[:k]


def _kuq(rng, k):
    from datasets import load_dataset
    d = list(load_dataset("amayuelas/KUQ", data_files="knowns_unknowns.jsonl", split="train"))
    rng.shuffle(d)
    pos, neg = [], []
    for it in d:
        if it.get("unknown") is None:
            continue
        abst = bool(it["unknown"])                 # unknown -> should abstain
        row = {"prompt": str(it["question"]).strip(), "should_abstain": abst, "dataset": "KUQ"}
        (pos if abst else neg).append(row)
        if len(pos) >= k and len(neg) >= k:
            break
    return pos[:k] + neg[:k]


def publish_abstentionbench(out_path: str, n: int = 300, seed: int = 42) -> dict:
    rng = random.Random(seed)
    k = max(2, n // 6)                              # k should_abstain + k should_answer per dataset (3 datasets)
    rows = _squad2(rng, k) + _selfaware(rng, k) + _kuq(rng, k)
    for i, r in enumerate(rows):
        r["id"] = f"{r['dataset']}-{i}"
    rng.shuffle(rows)
    rows = rows[:n]
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    from collections import Counter
    return {"total": len(rows),
            "dataset_dist": dict(Counter(r["dataset"] for r in rows)),
            "should_abstain_dist": dict(Counter(r["should_abstain"] for r in rows))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(publish_abstentionbench(a.out, a.n, a.seed), indent=1))
