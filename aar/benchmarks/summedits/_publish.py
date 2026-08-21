"""Publisher for summedits — builds secret_dir/summedits.jsonl from SummEdits
(Laban et al., EMNLP 2023; `Salesforce/summedits`, CC-BY-4.0, one `train` split of 6,348 rows;
label 1 = consistent, 0 = inconsistent; fields doc / summary / domain).

Disciplines (the 2026-06-17 audit):
  - **per-domain label-balanced** (≈15 consistent + 15 inconsistent × 10 domains) so the headline
    BALANCED ACCURACY is stable and a constant policy floors at 0.5 (the full set is 37.6/62.4).
  - **cap long documents** (drop docs > DOC_CHAR_CAP — billsum/podcast docs run 7–12k chars) so ≤7B
    context isn't exceeded; do NOT truncate (would drop the evidence and corrupt the gold).
Inconsistent items are atomic edits of a verified-consistent seed, so the two classes are surface-
identical — no length/lexical artifact (audited).

Item schema (one per line):
  {"prompt": <doc+summary+yes/no question>, "gold": "yes"|"no", "group": <domain>}
  (gold "yes" = the summary IS consistent with the document; single-token yes/no → clean logprob argmax.)

Usage (standalone, for validation):
  python -m aar.benchmarks.summedits._publish <out.jsonl> [--n 300] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DOC_CHAR_CAP = 12000   # ≈3k tokens — drop longer docs so ≤7B context isn't truncated
PROMPT = ('Document:\n{doc}\n\nSummary:\n{summary}\n\n'
          'Is the summary factually consistent with the document? '
          'Answer with exactly one word: "yes" or "no".')


def publish_summedits(out_path: str, n: int = 300, seed: int = 42) -> dict:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/summedits", split="train")
    rng = random.Random(seed)
    buckets: dict[tuple[str, int], list] = defaultdict(list)
    n_longdoc = 0
    for r in ds:
        doc, summ = r.get("doc"), r.get("summary")
        if not doc or not summ:
            continue
        if len(doc) > DOC_CHAR_CAP:
            n_longdoc += 1
            continue
        buckets[(str(r["domain"]), int(r["label"]))].append(r)

    groups = sorted({g for (g, _) in buckets})
    per_class = max(1, (n // max(1, len(groups))) // 2)   # ≈15 with 10 domains, n=300
    chosen: list[tuple[str, int, dict]] = []
    for g in groups:
        for lab in (1, 0):                # consistent, inconsistent
            pool = sorted(buckets.get((g, lab), []), key=lambda r: r["summary"])
            rng.shuffle(pool)
            chosen.extend((g, lab, r) for r in pool[:per_class])
    rng.shuffle(chosen)
    chosen = chosen[:n]

    rows = [{"prompt": PROMPT.format(doc=str(r["doc"]).strip(), summary=str(r["summary"]).strip()),
             "null_prompt": PROMPT.format(doc="N/A", summary=str(r["summary"]).strip()),
             "gold": "yes" if lab == 1 else "no", "group": g}
            for g, lab, r in chosen]
    Path(out_path).write_text("".join(json.dumps(x) + "\n" for x in rows))
    return {
        "total": len(rows), "dropped_long_docs": n_longdoc,
        "class_dist": dict(Counter(x["gold"] for x in rows)),
        "group_dist": dict(Counter(x["group"] for x in rows)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(publish_summedits(a.out, a.n, a.seed), indent=1))
