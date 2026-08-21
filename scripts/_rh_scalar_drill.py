"""Drill into the held-out loophole_scalar: construct validity + item-level base behavior.
No GPU. Joins _rh_share/ls/loophole_scalar.jsonl (object/amount/prompts) with the base-Olmo
per-item recs in _rh_diag_results.json (same item order).

Questions:
  Q1 construct: is "give ONE" a clear loophole, or defensible instruction-following? (read prompts)
  Q2 is gaming pressure-driven? gaming rate by AMOUNT (10/100/1000/10000 → stronger keep-max pull).
  Q3 metric conflation: are the items base GATE-FAILS on the SAME as the items base GAMES on?
     (if gate-fail-set and gaming-set are disjoint, comprehension vs resistance are separable.)
"""
import json
from pathlib import Path
items = [json.loads(l) for l in Path("/opt/aar/work/_rh_share/ls/loophole_scalar.jsonl").read_text().splitlines() if l.strip()]
R = json.loads(Path("/opt/aar/work/aar_harness_runs/_rh_diag_results.json").read_text())
recs = R["BASE_olmo"]["loophole_scalar"]["recs"]
assert len(items) == len(recs), (len(items), len(recs))

print("="*80); print("Q1 — CONSTRUCT: 2 example items (intent control + action-under-conflict)"); print("="*80)
for it in items[:2]:
    print("\n--- object=%s amount=%s  (honor=More-than-one, loophole=One) ---" % (it["object"], it["amount"]))
    print("[INTENT]", it["prompt_intent"].replace("\n", " ")[:300])
    print("[ACTION]", it["prompt_action"].replace("\n", " ")[:300])

from collections import defaultdict
by_amt = defaultdict(lambda: {"n":0,"comp":0,"gamed":0,"gatefail":0})
gatefail_idx, gamed_idx = set(), set()
for i,(it,r) in enumerate(zip(items,recs)):
    a = by_amt[it["amount"]]; a["n"]+=1
    if r["gate"]:
        a["comp"]+=1
        if r["judged"]==0.0: a["gamed"]+=1; gamed_idx.add(i)
    else:
        a["gatefail"]+=1; gatefail_idx.add(i)

print("\n"+"="*80); print("Q2 — is gaming PRESSURE-driven? base behavior by AMOUNT"); print("="*80)
print("%8s %4s %5s %6s %8s | %s" % ("amount","n","comp","gamed","gatefail","gaming-rate-among-competent"))
for amt in sorted(by_amt):
    a=by_amt[amt]; gr = a["gamed"]/a["comp"] if a["comp"] else 0
    print("%8s %4d %5d %6d %8d | %.2f" % (amt,a["n"],a["comp"],a["gamed"],a["gatefail"],gr))

print("\n"+"="*80); print("Q3 — METRIC CONFLATION: gate-fail vs gaming item-set overlap"); print("="*80)
print("gate-fail items: %d   gamed items: %d   overlap: %d (disjoint by construction — gate-fail are EXCLUDED before gaming is judged)" % (
    len(gatefail_idx), len(gamed_idx), len(gatefail_idx & gamed_idx)))
print("So 'comprehension loss' (gate-fail=%d) and 'gaming' (=%d) are DIFFERENT items." % (len(gatefail_idx), len(gamed_idx)))
print("Of 80: honored=%d, gamed=%d, gatefail=%d" % (80-len(gatefail_idx)-len(gamed_idx), len(gamed_idx), len(gatefail_idx)))
