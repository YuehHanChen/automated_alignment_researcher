#!/usr/bin/env python3
"""Root-cause: decompose the bias team's top-scored submissions per benchmark.

For each submission we have closed_pct on the 3 scored legs (bias_mt_decision, bias_mt_occupation,
bbq) and on the held-out (bbq_heldout). Two hypotheses to separate:
  (A) FORMAT mismatch: scored gain rides the MT free-form legs; bbq (MCQ) ~ bbq_heldout (MCQ) flat.
  (B) CATEGORY overfit: scored bbq (MCQ) IMPROVES but bbq_heldout (MCQ, physical_appearance) does not.
The bbq vs bbq_heldout column pair is the discriminator (same format, different category)."""
import json, glob, os, re, statistics

TEAM = "bias-olmo-opus48-20260628-055137"
D = f"/opt/aar/eval-user/aar_teams/{TEAM}/heldout_scores"
LEGS = ["bias_mt_decision", "bias_mt_occupation", "bbq", "bbq_heldout"]


def method_of(fn):
    m = re.search(r"__([a-z0-9_]+)-\d+\.json$", fn)
    return m.group(1) if m else fn[:24]


rows = []
for f in glob.glob(os.path.join(D, "*.json")):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    pb = j.get("per_benchmark", {})
    head = j.get("headline_pct")
    pf = j.get("passes_filter")
    cl = {leg: (pb.get(leg, {}) or {}).get("closed_pct") for leg in LEGS}
    mn = {leg: (pb.get(leg, {}) or {}).get("mean") for leg in LEGS}
    if head is None:
        continue
    rows.append({"m": method_of(os.path.basename(f)), "head": head, "pf": pf, "cl": cl, "mn": mn})

rows.sort(key=lambda r: r["head"], reverse=True)
print(f"bias team: {len(rows)} submissions   (closed_pct per benchmark; bbq vs bbq_heldout = same MCQ format, diff category)")
print(f"{'method':30} {'pf':3} {'SCORED':>7} | {'mt_dec':>7} {'mt_occ':>7} {'bbq':>7} || {'HELDOUT':>8}")
print("-" * 92)
for r in rows[:18]:
    c = r["cl"]
    def g(x): return f"{x:+.1f}" if isinstance(x, (int, float)) else "  ?  "
    print(f"{r['m'][:30]:30} {str(bool(r['pf']))[0]:3} {r['head']:+7.1f} | "
          f"{g(c['bias_mt_decision']):>7} {g(c['bias_mt_occupation']):>7} {g(c['bbq']):>7} || {g(c['bbq_heldout']):>8}")

# aggregate correlation signals
pf_rows = [r for r in rows if r["pf"]]
def col(leg): return [r["cl"][leg] for r in pf_rows if isinstance(r["cl"][leg], (int, float)) and isinstance(r["cl"]["bbq_heldout"], (int, float))]
def hocol(leg): return [r["cl"]["bbq_heldout"] for r in pf_rows if isinstance(r["cl"][leg], (int, float)) and isinstance(r["cl"]["bbq_heldout"], (int, float))]

def corr(xs, ys):
    if len(xs) < 5: return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = (sum((x-mx)**2 for x in xs))**.5; sy = (sum((y-my)**2 for y in ys))**.5
    return cov/(sx*sy) if sx and sy else None

print("\n--- which scored leg best predicts the held-out? (corr of leg closed% vs bbq_heldout closed%, filter-passing) ---")
for leg in ["bias_mt_decision", "bias_mt_occupation", "bbq"]:
    xs, ys = col(leg), hocol(leg)
    c = corr(xs, ys)
    print(f"  corr({leg:18} , bbq_heldout) = {c:+.3f}   (n={len(xs)})" if c is not None else f"  {leg}: n<5")

# scored-bbq vs held-out-bbq: the category test
both = [(r["cl"]["bbq"], r["cl"]["bbq_heldout"]) for r in pf_rows
        if isinstance(r["cl"]["bbq"], (int, float)) and isinstance(r["cl"]["bbq_heldout"], (int, float))]
hi_bbq = [hb for sb, hb in both if sb > 15]
print(f"\n--- CATEGORY test: submissions that clearly improved SCORED bbq (>+15%): {len(hi_bbq)}")
if hi_bbq:
    print(f"    their bbq_heldout closed%: min={min(hi_bbq):+.1f} median={statistics.median(hi_bbq):+.1f} max={max(hi_bbq):+.1f}")
    print(f"    => if these are flat/negative, the de-biasing is CATEGORY-SPECIFIC (overfit to scored bbq categories)")
