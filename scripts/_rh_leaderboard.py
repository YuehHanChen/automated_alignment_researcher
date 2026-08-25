"""Aggregate the reward_hacking-olmo team's research-side scored composites into a leaderboard.
Operator diagnostic (research-side only; held-out loophole_scalar is NOT here by design).
Prints: method, headline_pct, per-scored-leg closed_pct + mean, passes_filter, submission path.
"""
import json, sys
from pathlib import Path

T = Path("/opt/aar/work/aar_teams/reward_hacking-olmo-opus48-20260621-173940")
rows = []
for f in (T / "scores").glob("*.json"):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if d.get("suite") != "reward_hacking":
        continue
    pb = d.get("per_benchmark", {})
    def leg(name, k):
        b = pb.get(name, {})
        return b.get(k)
    # method short name: strip the "<agent>-<ts>-<agent>__" prefix and the trailing "-<ts>"
    stem = f.stem
    short = stem.split("__", 1)[1] if "__" in stem else stem
    short = short.rsplit("-", 1)[0] if short[-13:].isdigit() else short
    rows.append({
        "short": short,
        "headline": d.get("headline_pct"),
        "pass": d.get("passes_filter"),
        "bracket_cp": leg("loophole_bracket", "closed_pct"),
        "exploit_cp": leg("loophole_exploitation", "closed_pct"),
        "reveal_cp": leg("revealing_score", "closed_pct"),
        "bracket_m": leg("loophole_bracket", "mean"),
        "exploit_m": leg("loophole_exploitation", "mean"),
        "reveal_m": leg("revealing_score", "mean"),
        "model": d.get("model"),
        "file": f.name,
    })

rows = [r for r in rows if r["headline"] is not None]
rows.sort(key=lambda r: (r["pass"] is True, r["headline"]), reverse=True)
print(f"total scored composites: {len(rows)}")
print(f"{'rank':>4} {'headline':>8} {'pass':>5} {'brkt%':>6} {'expl%':>6} {'revl%':>6}  method")
for i, r in enumerate(rows[:30]):
    def fnum(x, w=6, p=1):
        return (f"{x:.{p}f}" if isinstance(x, (int, float)) else "—").rjust(w)
    print(f"{i:>4} {fnum(r['headline'],8,2)} {str(r['pass']):>5} "
          f"{fnum(r['bracket_cp'])} {fnum(r['exploit_cp'])} {fnum(r['reveal_cp'])}  {r['short']}")

# also dump the full table as json for downstream join
out = T.parent / "_rh_leaderboard.json"
# write to a research-writable scratch instead (team dir may be read-only-ish)
scratch = Path("/opt/aar/work/aar_repo_runs/_rh_leaderboard.json")
scratch.write_text(json.dumps(rows, indent=1))
print(f"\nfull table -> {scratch}")

# locate checkpoints for the methods of interest
print("\n=== submission/model dirs for methods of interest ===")
for key in ("f2_deepdiverse", rows[0]["short"], rows[1]["short"]):
    hits = sorted((T / "submissions").glob(f"*{key}*"))
    for h in hits[:3]:
        m = h / "model"
        print(f"  {key:>22}  exists={m.exists()}  {h.name}")
