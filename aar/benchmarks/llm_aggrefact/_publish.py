"""Publisher for llm_aggrefact — builds secret_dir/llm_aggrefact.jsonl from LLM-AggreFact
(Tang et al., MiniCheck, EMNLP 2024). The official `lytang/LLM-AggreFact` is GATED (request
access for production); we fall back to the ungated pre-08.09.2024 mirror
`NinaCalvi/llm_aggrefact_pre_aug9` (10 constituents, no RAGTruth) — both share the
`dataset / doc / claim / label` schema (label 1 = supported).

Two REQUIRED disciplines (BENCHMARK_CATALOG factuality refocus + the 2026-06-17 audit):
  - **EXCLUDE `dataset == "RAGTruth"`** — it is the 11th constituent and would overlap our
    existing `ragtruth` leg; dropping it preserves item-independence.
  - **cap long documents** (drop docs > DOC_CHAR_CAP; do NOT truncate — truncation can remove the
    evidence a "supported" claim relies on and corrupt the gold) so ≤7B context isn't exceeded.
The subset is **per-constituent label-balanced** (≈15 supported + 15 unsupported × 10 constituents)
so the headline BALANCED ACCURACY is stable and a constant policy floors at 0.5.

Item schema (one per line):
  {"prompt": <doc+claim+yes/no question>, "gold": "yes"|"no", "group": <constituent>}
  (gold "yes" = the claim IS supported by the document; single-token yes/no → clean logprob argmax.)

Usage (standalone, for validation):
  python -m aar.benchmarks.llm_aggrefact._publish <out.jsonl> [--n 300] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DOC_CHAR_CAP = 12000   # ≈3k tokens — drop longer docs so ≤7B context isn't truncated
PROMPT = ('Document:\n{doc}\n\nClaim: {claim}\n\n'
          'Is the claim fully supported by the document? '
          'Answer with exactly one word: "yes" or "no".')

# official (gated) first, then the ungated pre-aug9 mirror (no RAGTruth, same schema).
_SOURCES = [("lytang/LLM-AggreFact", "test"), ("NinaCalvi/llm_aggrefact_pre_aug9", "test")]


def _load_rows():
    from datasets import load_dataset
    last = None
    for repo, split in _SOURCES:
        try:
            return repo, load_dataset(repo, split=split)
        except Exception as e:   # gated 403 / missing -> try the next source
            last = e
    raise RuntimeError(f"could not load LLM-AggreFact (official gated? mirror missing?): {last}")


def publish_llm_aggrefact(out_path: str, n: int = 300, seed: int = 42, group_filter=None) -> dict:
    repo, ds = _load_rows()
    rng = random.Random(seed)
    buckets: dict[tuple[str, int], list] = defaultdict(list)
    n_ragtruth = n_longdoc = 0
    for r in ds:
        dset = str(r["dataset"])
        if dset == "RAGTruth":            # item-independence from our ragtruth leg
            n_ragtruth += 1
            continue
        if group_filter is not None and dset not in group_filter:   # A/B source-split (faithfulness suite)
            continue
        doc, claim = r.get("doc"), r.get("claim")
        if not doc or not claim:
            continue
        if len(doc) > DOC_CHAR_CAP:       # drop (don't truncate -> would corrupt gold)
            n_longdoc += 1
            continue
        buckets[(dset, int(r["label"]))].append(r)

    groups = sorted({g for (g, _) in buckets})
    per_class = max(1, (n // max(1, len(groups))) // 2)   # ≈15 with 10 groups, n=300
    chosen: list[tuple[str, int, dict]] = []
    for g in groups:
        for lab in (1, 0):                # supported, unsupported
            pool = sorted(buckets.get((g, lab), []), key=lambda r: r["claim"])
            rng.shuffle(pool)
            chosen.extend((g, lab, r) for r in pool[:per_class])
    rng.shuffle(chosen)
    chosen = chosen[:n]

    rows = [{"prompt": PROMPT.format(doc=str(r["doc"]).strip(), claim=str(r["claim"]).strip()),
             "null_prompt": PROMPT.format(doc="N/A", claim=str(r["claim"]).strip()),
             "gold": "yes" if lab == 1 else "no", "group": g}
            for g, lab, r in chosen]
    Path(out_path).write_text("".join(json.dumps(x) + "\n" for x in rows))
    return {
        "total": len(rows), "source": repo,
        "excluded_ragtruth": n_ragtruth, "dropped_long_docs": n_longdoc,
        "class_dist": dict(Counter(x["gold"] for x in rows)),
        "group_dist": dict(Counter(x["group"] for x in rows)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(publish_llm_aggrefact(a.out, a.n, a.seed), indent=1))
