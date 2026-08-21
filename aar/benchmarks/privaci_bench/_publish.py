"""Publisher for privaci_bench — builds secret_dir/privaci_bench.jsonl from the PrivaCI-Bench
cases (`HKUST-KnowComp/PrivaCI-Bench`, `HF_cache/cases` via `datasets.load_from_disk`; the
.arrow cases are committed in the repo, so a shallow clone has them).

The regulatory COMPLIANCE task on the **non-GDPR remainder** (AI_ACT + HIPAA + ACLU); GDPR is held
out (→ privaci_gdpr_heldout). The source prompt (`prompts/direct_answer_prompt.txt`) and the
source norm_type→class map (`label_transform`: prohibit→negative, permit→positive, not
applicable→same) are read/applied verbatim — no authored prompts. **2026-06-24 REDESIGN: scored by
GUARDED PROHIBIT-RECALL** (see benchmark.py `_guarded_prohibit_recall`; supersedes 06-23 2-class
macro-F1). "not-applicable" is a per-regulation data artifact (GDPR & ACLU have 0) — not a
transferable construct — so it is not scored (predicting it is a clean miss). prohibit-recall is
class-prior-invariant, so the scored set + the prohibit-skewed GDPR held-out share an optimum.

Two disciplines on top of the raw cases (BENCHMARK_CATALOG §7 + README ≤300-item rule):

  - **article-id-leak filter** — drop cases whose `case_content` prints a gold article/recital/
    section id verbatim. Measured leak rates: AI_ACT 9.9%, HIPAA 4.7%, ACLU 34.8% (GDPR 0.6%).
    Those cases let a model pattern-match the regulation id instead of reasoning about contextual
    integrity; dropping them keeps the task CI-*reasoning*, not string-matching.
  - **class-balanced (2-class), AI-Act-anchored, ≤300** — a frozen seeded stratified sample of
    ~`per_class` = n//2 items per scored class (prohibit/permit; "not-applicable" no longer sampled).
    AI_ACT is the balanced anchor; HIPAA + ACLU add cross-regulation diversity via small per-class
    quotas, with AI_ACT backfilling any scarce-domain shortfall. (Under prohibit-recall the scored
    balance no longer carries the Goodhart guard — the permit-recall floor does — but a balanced set
    keeps both per-class recalls well-estimated; constant always-prohibit → 0, always-permit → 0.)

Item schema (one per line):
  {"prompt": <filled direct_answer_prompt>, "gold": "negative"|"positive"|"not applicable", "domain": str}

Usage (standalone, for validation):
  python -m aar.benchmarks.privaci_bench._publish <repo_dir> <out.jsonl> [--n 300] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

# Non-GDPR remainder (GDPR held out → privaci_gdpr_heldout). AI_ACT is the balanced anchor.
DOMAINS = ["AI_ACT", "HIPAA", "ACLU"]

# norm_type → compliance class (source label_transform). Synonyms kept for robustness.
NORM2CLASS = {
    "prohibit": "negative", "prohibited": "negative", "negative": "negative",
    "permit": "positive", "permitted": "positive", "positive": "positive",
    "not applicable": "not applicable",
}

# Per-(scarce-domain, class) caps. AI_ACT backfills each class to `per_class`, so it stays the
# anchor (~70%) while HIPAA + ACLU contribute cross-regulation diversity. ACLU has no N-A class.
QUOTA = {
    ("HIPAA", "negative"): 15, ("HIPAA", "positive"): 20, ("HIPAA", "not applicable"): 25,
    ("ACLU", "negative"): 20, ("ACLU", "positive"): 10, ("ACLU", "not applicable"): 0,
}

_ARTID = re.compile(r"(article|recital|section)\s*[0-9]+[0-9a-z.()]*", re.I)


def _leaks_article_id(case: dict) -> bool:
    """True iff the case_content prints a gold article/recital/section id verbatim."""
    arts = (case.get("followed_articles") or []) + (case.get("violated_articles") or [])
    txt = case["case_content"].lower()
    for a in arts:
        a = str(a).strip().lower()
        if not a:
            continue
        if a in txt:
            return True
        m = _ARTID.search(a)
        if m and m.group(0) in txt:
            return True
    return False


def publish_privaci_bench(repo_dir: str, out_path: str, n: int = 300, seed: int = 42) -> dict:
    # `n` is the total ≤300 cap (README rule 7 — same signature contract as every other publisher,
    # so publish_suite's EVAL_MAX_N enforcement + `--force-n` re-baseline prep + subset_size all
    # apply uniformly). 2026-06-23 redesign: the scored set is balanced over the TWO classes the
    # held-out GDPR actually contains (prohibit/permit), so per_class = n // 2. "not applicable" is
    # NOT sampled — it is a per-domain data artifact (GDPR/ACLU have none), so scoring it made the
    # scored headroom an off-target AI-Act scope-recognition skill that couldn't transfer to GDPR.
    per_class = n // 2
    from datasets import load_from_disk

    repo = Path(repo_dir)
    template = (repo / "prompts" / "direct_answer_prompt.txt").read_text()
    cases = load_from_disk(str(repo / "HF_cache" / "cases"))

    # Bucket the leak-filtered cases by (domain, class).
    buckets: dict[tuple[str, str], list[dict]] = {}
    for dom in DOMAINS:
        for case in cases[dom]:
            if _leaks_article_id(case):
                continue
            cls = NORM2CLASS.get(str(case["norm_type"]).strip().lower())
            if cls is None:
                continue
            buckets.setdefault((dom, cls), []).append(case)

    rng = random.Random(seed)
    chosen: list[tuple[str, str, dict]] = []     # (domain, class, case)
    realized: Counter[tuple[str, str]] = Counter()
    for cls in CLASSES_ORDER:
        picked: list[tuple[str, dict]] = []
        # scarce domains first (capped by QUOTA, then by availability)
        for dom in ("HIPAA", "ACLU"):
            pool = sorted(buckets.get((dom, cls), []), key=lambda c: c["case_content"])
            rng.shuffle(pool)
            take = min(QUOTA.get((dom, cls), 0), len(pool))
            for c in pool[:take]:
                picked.append((dom, c))
                realized[(dom, cls)] += 1
        # AI_ACT backfills to reach per_class exactly
        pool = sorted(buckets.get(("AI_ACT", cls), []), key=lambda c: c["case_content"])
        rng.shuffle(pool)
        take = min(max(per_class - len(picked), 0), len(pool))
        for c in pool[:take]:
            picked.append(("AI_ACT", c))
            realized[("AI_ACT", cls)] += 1
        chosen.extend((dom, cls, c) for dom, c in picked)

    rows = [{"prompt": template.format(domain=dom, event=case["case_content"]),
             "gold": cls, "domain": dom} for dom, cls, case in chosen]
    rng.shuffle(rows)   # mix domains/classes in file order (no positional shortcut)
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))

    return {
        "total": len(rows),
        "class_dist": dict(Counter(r["gold"] for r in rows)),
        "domain_dist": dict(Counter(r["domain"] for r in rows)),
        "by_domain_class": {f"{d}/{c}": realized[(d, c)] for (d, c) in sorted(realized)},
    }


# Scored-set sampler classes — prohibit/permit only (the label space shared with the GDPR held-out);
# "not applicable" dropped (2026-06-23 redesign, see per_class note + benchmark.py SCORED_CLASSES).
CLASSES_ORDER = ["negative", "positive"]


def publish_privaci_gdpr_heldout(repo_dir: str, out_path: str, n: int = 300, seed: int = 42) -> dict:
    """Build the GDPR HELD-OUT generalization probe (privaci_gdpr_heldout) from the same
    PrivaCI-Bench cases, scored by the IDENTICAL guarded-prohibit-recall scorer as privaci_bench.

    Two deliberate differences from `publish_privaci_bench` (BENCHMARK_CATALOG §7):
      - **GDPR only** (the domain held OUT of the scored non-GDPR remainder) → cross-regulation OOD.
      - **NATURAL class distribution, NOT balanced.** GDPR is the skewed/most-memorized regime
        (≈78.5% "prohibit"); we keep that distribution on purpose — prohibit-recall is prior-invariant
        (guarded by a permit-recall floor so always-"prohibit" → 0), so the skew is scored honestly,
        never raw accuracy.
    The article-id-leak filter is applied identically (so the task is CI-reasoning, not id string-
    matching), and the same verbatim source prompt is used. `n` is the ≤300 cap (README rule 7),
    a frozen seeded subsample of the leak-filtered GDPR cases (a held-out isn't optimized →
    directional power is enough). Schema is identical to privaci_bench."""
    from datasets import load_from_disk

    repo = Path(repo_dir)
    template = (repo / "prompts" / "direct_answer_prompt.txt").read_text()
    cases = load_from_disk(str(repo / "HF_cache" / "cases"))

    pool: list[tuple[dict, str]] = []
    for case in cases["GDPR"]:
        if _leaks_article_id(case):
            continue
        cls = NORM2CLASS.get(str(case["norm_type"]).strip().lower())
        if cls is None:
            continue
        pool.append((case, cls))

    pool.sort(key=lambda x: x[0]["case_content"])   # deterministic order before the seeded draw
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = pool[:n]                               # natural distribution preserved (no per-class quota)

    rows = [{"prompt": template.format(domain="GDPR", event=case["case_content"]),
             "gold": cls, "domain": "GDPR"} for case, cls in chosen]
    rng.shuffle(rows)
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))

    return {
        "total": len(rows),
        "pool_after_leak_filter": len(pool),
        "class_dist": dict(Counter(r["gold"] for r in rows)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_dir")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300, help="total ≤300 cap (2-class balanced; per_class=n//2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gdpr-heldout", action="store_true",
                    help="build the GDPR held-out probe (natural distribution) instead of the scored remainder")
    a = ap.parse_args()
    fn = publish_privaci_gdpr_heldout if a.gdpr_heldout else publish_privaci_bench
    print(json.dumps(fn(a.repo_dir, a.out, a.n, a.seed), indent=1))
