#!/usr/bin/env python3
import json, os, collections

HD = "/opt/aar/eval-user/holdout/olmo/bias"
for name in ("bbq", "bbq_heldout"):
    rows = [json.loads(l) for l in open(os.path.join(HD, name + ".jsonl"))]
    cond = collections.Counter(r["cond"] for r in rows)
    grp = collections.Counter(r["group"] for r in rows)
    # sanity: on ambig is correct==unk? on disambig is correct!=unk?
    ambig_unk = sum(1 for r in rows if r["cond"] == "ambig" and r["correct"] == r["unk"])
    disambig_notunk = sum(1 for r in rows if r["cond"] == "disambig" and r["correct"] != r["unk"])
    nA = cond.get("ambig", 0); nD = cond.get("disambig", 0)
    print(f"{name}: n={len(rows)}  ambig={nA} disambig={nD}")
    print(f"   groups={dict(grp)}")
    print(f"   ambig items where correct==unk: {ambig_unk}/{nA}   disambig items where correct!=unk: {disambig_notunk}/{nD}")
    print(f"   => on disambig, answering 'unk' (over-abstain) is WRONG for {disambig_notunk} items")
