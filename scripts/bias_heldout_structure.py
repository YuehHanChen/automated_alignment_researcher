#!/usr/bin/env python3
"""Probe bbq vs bbq_heldout structure to test the over-abstention / category-gap hypothesis."""
import json, glob, os, collections

HS = "/opt/aar/eval-user/aar_teams/bias-olmo-opus48-20260628-055137/heldout_scores"
HD = "/opt/aar/eval-user/holdout/olmo/bias"

# 1) full per_benchmark entry for bbq + bbq_heldout on the top-scored submission
best = None
for f in glob.glob(HS + "/*.json"):
    j = json.load(open(f))
    if j.get("headline_pct") is not None and (best is None or j["headline_pct"] > best[0]):
        best = (j["headline_pct"], f, j)
h, f, j = best
print("=== TOP-scored submission:", os.path.basename(f)[:62], "(scored", f"{h:+.1f}%)")
pb = j["per_benchmark"]
for k in ("bbq", "bbq_heldout"):
    print(f"  {k}: {json.dumps(pb.get(k, {}))}")

# 2) structure of the two jsonls: fields, categories, ambiguous/disambiguated, unknown option
print("\n=== jsonl structure / category composition ===")
for name in ("bbq", "bbq_heldout"):
    p = os.path.join(HD, name + ".jsonl")
    if not os.path.exists(p):
        print(f"  {name}: MISSING {p}"); continue
    rows = [json.loads(l) for l in open(p)]
    keys = sorted(rows[0].keys())
    cats = collections.Counter(str(r.get("category", r.get("cat", "?"))) for r in rows)
    cond = collections.Counter()
    for r in rows:
        for ck in ("context_condition", "condition", "ambig", "context_type"):
            if ck in r:
                cond[f"{ck}={r[ck]}"] += 1
                break
    print(f"  {name}: n={len(rows)}  fields={keys}")
    print(f"    categories: {dict(cats)}")
    print(f"    context_condition: {dict(cond) or 'NONE FOUND'}")
    # show one sample item (truncated)
    s = rows[0]
    samp = {kk: (str(s[kk])[:90] if isinstance(s[kk], str) else s[kk]) for kk in s}
    print(f"    sample[0]: {json.dumps(samp)[:400]}")
