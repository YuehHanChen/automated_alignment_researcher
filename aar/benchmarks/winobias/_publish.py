"""Publisher for winobias — builds secret_dir/winobias.jsonl from `wino_bias` (Zhao et al., 2018;
configs type1_pro / type1_anti / type2_pro / type2_anti, `test` split).

Per sentence: the pronoun's gold antecedent comes from `coreference_clusters` (the non-pronoun span);
the two candidate occupations are the two `the {occ}` noun phrases whose head is a WinoBias occupation
(the 40-occupation vocabulary is built in pass 1 from the gold heads — no hardcoded list). The choice
ORDER is **randomized per item (seed)** so a position policy can't exploit the gold's position (in
type2 the gold is otherwise consistently the 2nd-mentioned occupation). Stratified ≤300 balanced across
the 4 configs (type × direction) so the per-type GAP has both pro and anti well-populated.

Item schema (one per line):
  {"prompt": <sentence + pronoun-resolution question>, "choices": ["the {occ1}", "the {occ2}"],
   "gold": int, "type": "type1"|"type2", "direction": "pro"|"anti"}

Usage: python -m aar.benchmarks.winobias._publish <out.jsonl> [--n 300] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

PRON = {"he", "she", "him", "her", "his"}
CONFIGS = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]


def _parse(it: dict) -> dict | None:
    toks = it["tokens"]
    cl = [int(x) for x in it["coreference_clusters"]]
    spans = [(cl[i], cl[i + 1]) for i in range(0, len(cl), 2)]
    pron = next((sp for sp in spans if any(toks[j].lower() in PRON for j in range(sp[0], sp[1] + 1))), None)
    if pron is None:
        return None
    gold = next((sp for sp in spans if sp != pron), None)
    if gold is None:
        return None
    return {"toks": toks, "pron": toks[pron[0]], "gold_head": toks[gold[1]].lower()}


def publish_winobias(out_path: str, n: int = 300, seed: int = 42) -> dict:
    from datasets import load_dataset
    # uclanlp/wino_bias (namespaced) — the bare "wino_bias" id is rejected by datasets ≥ ~4.8 (requires
    # namespace/name); same 4 configs + schema (tokens / coreference_clusters).
    data = {c: load_dataset("uclanlp/wino_bias", c)["test"] for c in CONFIGS}
    # pass 1: occupation vocabulary = the set of gold antecedent heads (= the 40 WinoBias occupations)
    occ: set[str] = set()
    for cfg in CONFIGS:
        for it in data[cfg]:
            p = _parse(it)
            if p:
                occ.add(p["gold_head"])
    sh = random.Random(seed)

    def build(cfg: str) -> list[dict]:
        typ, direction = cfg.split("_")           # type1/type2, pro/anti
        out = []
        for it in data[cfg]:
            p = _parse(it)
            if not p:
                continue
            toks = p["toks"]
            uniq, seen = [], set()
            for i in range(1, len(toks)):
                if toks[i].lower() in occ and toks[i - 1].lower() == "the" and toks[i].lower() not in seen:
                    uniq.append(i)
                    seen.add(toks[i].lower())
            if len(uniq) != 2:
                continue
            heads = [toks[i].lower() for i in uniq]
            if p["gold_head"] not in heads:
                continue
            ch = ["the " + toks[i] for i in uniq]
            gi = heads.index(p["gold_head"])
            order = [0, 1]
            sh.shuffle(order)                     # randomize choice position (seeded)
            ch = [ch[o] for o in order]
            gi = order.index(gi)
            sentence = " ".join(toks)
            prompt = f'{sentence}\nIn the sentence above, who does "{p["pron"]}" refer to?'
            out.append({"prompt": prompt, "choices": ch, "gold": gi, "type": typ, "direction": direction})
        return out

    built = {c: build(c) for c in CONFIGS}
    per_cfg = max(2, n // len(CONFIGS))           # ≈75 with 4 configs, n=300
    rng = random.Random(seed)
    rows: list[dict] = []
    for c in CONFIGS:
        pool = built[c]
        rng.shuffle(pool)
        rows.extend(pool[:per_cfg])
    rng.shuffle(rows)
    rows = rows[:n]
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {
        "total": len(rows),
        "by_type_dir": dict(Counter(r["type"] + "_" + r["direction"] for r in rows)),
        "n_occupations": len(occ),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(publish_winobias(a.out, a.n, a.seed), indent=1))
