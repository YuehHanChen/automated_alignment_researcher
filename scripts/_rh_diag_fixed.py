"""Fixed-subset transfer check: remove the competence-gate-composition confound.
recs are in item order and identical across models, so rec[i] is the same item.
For each leg, fix the item set to {i : BASE gate passes}; report each model's
ACTION honor-rate (action==honor) on that fixed set -> isolates behavior change
from gate attrition.
"""
import json
from pathlib import Path
R = json.loads(Path("/opt/aar/work/aar_harness_runs/_rh_diag_results.json").read_text())
LEGS = ["loophole_bracket", "loophole_exploitation", "revealing_score", "loophole_scalar"]
base = "BASE_olmo"
models = list(R.keys())

print("ACTION honor-rate on the FIXED base-gated item set (same denominator for all models)")
hdr = f"{'model':28}" + "".join(f"{l.replace('loophole_','lp_')[:13]:>15}" for l in LEGS)
print(hdr)
base_idx = {l: [i for i, r in enumerate(R[base][l]["recs"]) if r["gate"]] for l in LEGS}
for m in models:
    row = f"{m[:28]:28}"
    for l in LEGS:
        idx = base_idx[l]
        recs = R[m][l]["recs"]
        hr = sum(1 for i in idx if recs[i]["action"] == recs[i]["honor"]) / len(idx) if idx else 0.0
        row += f"{f'{hr:.3f}({len(idx)})':>15}"
    print(row)
print("\nDELTA vs base (fixed set):")
for m in models:
    if m == base: continue
    row = f"{m[:28]:28}"
    for l in LEGS:
        idx = base_idx[l]
        rb, rm = R[base][l]["recs"], R[m][l]["recs"]
        hb = sum(1 for i in idx if rb[i]["action"] == rb[i]["honor"]) / len(idx)
        hm = sum(1 for i in idx if rm[i]["action"] == rm[i]["honor"]) / len(idx)
        row += f"{f'{hm-hb:+.3f}':>15}"
    print(row)
