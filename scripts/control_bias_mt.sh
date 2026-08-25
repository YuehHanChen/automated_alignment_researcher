#!/bin/bash
#SBATCH --job-name=biasctrl
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/opt/aar/eval-user/biasctrl_%j.out
#
# VALIDITY AUDIT for the bias multi-turn legs (bias_mt_occupation). Two things:
#  (1) DUMP per-pair (final reply A, final reply B, invariance verdict) so a human can read the pairs the
#      judge flagged BIASED (different decision) vs NOT-BIASED (same) and check the LLM judge is correct.
#  (2) PLACEBO control: counterfactual (man vs woman) vs control (two different men). If invariance(cf) <<
#      invariance(control), the gender swap (not noise) drives non-invariance. Same temp-1/seed-1234 decoding.
R=/opt/aar/aar_repo
PY=/opt/aar/work/git/python
export PYTHONPATH="${R}"
export HF_HOME="${HF_HOME:-/opt/aar/eval-user/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/aar/eval-user/hf_datasets}"
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_GPUS=auto
export JUDGE_BACKEND=anthropic JUDGE_MODEL=claude-haiku-4-5 JUDGE_CONCURRENCY=100
for _ak in ANTHROPIC_API_KEY ANT_high_prio_API ANT_API_KEY; do
  _av="$(grep -m1 "^${_ak}=" /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
  [ -n "${_av}" ] && export "${_ak}=${_av}"
done
set -euo pipefail
OUT=/opt/aar/eval-user/biasctrl_records.json
echo "[biasctrl] $(date)"
${PY} - "$OUT" <<'PY'
import json, os, sys, tempfile
from aar.eval_pod.models import load_model
from aar.eval_pod.judges import make_anthropic_judge
from aar.benchmarks.bias_mt_occupation.benchmark import BiasMTOccupation
from aar.benchmarks.bias_mt_occupation._publish import OCCUPATIONS, MALE, FEMALE, _turns

OUTPATH = sys.argv[1]
K = 40  # pairs per arm

def eps(arm):
    out = []
    for p in range(K):
        occ = OCCUPATIONS[p % len(OCCUPATIONS)]
        if arm == "cf":
            (nA, gA), (nB, gB) = (MALE[p % len(MALE)], "male"), (FEMALE[p % len(FEMALE)], "female")
        else:
            (nA, gA), (nB, gB) = (MALE[p % len(MALE)], "male"), (MALE[(p + 7) % len(MALE)], "male")
        out.append({"pair_id": f"{arm}{p}", "variant": "A", "name": nA, "group": gA, "occupation": occ, "turns": _turns(nA, occ)})
        out.append({"pair_id": f"{arm}{p}", "variant": "B", "name": nB, "group": gB, "occupation": occ, "turns": _turns(nB, occ)})
    return out

class _Spec:
    id = "bias_mt_occupation"; name = "bias_mt_occupation"; subset_size = None

print("[biasctrl] loading model + judge...", flush=True)
model = load_model("allenai/Olmo-3-7B-Instruct")
model.apply_decoding(temperature=1.0, top_p=1.0, seed=1234)
judge = make_anthropic_judge(model="claude-haiku-4-5")
bench = BiasMTOccupation(_Spec(), tempfile.mkdtemp(), judge_fn=judge)

def run(arm, save_records=False):
    elist = eps(arm)
    trs = bench.rollout_batch(model, elist)
    pairs = {}
    for e, tr in zip(elist, trs):
        pairs.setdefault(e["pair_id"], {})[e["variant"]] = {"resp": bench._final(tr), "scene": e["turns"][0]["prompt"],
                                                            "name": e["name"], "occ": e["occupation"]}
    n_eng = n_inv = n = 0
    recs = []
    for pid, pv in pairs.items():
        if "A" not in pv or "B" not in pv:
            continue
        ra, rb = pv["A"]["resp"], pv["B"]["resp"]
        ea, sa = bench._verdict(ra)   # (engaged, sign) — verdict extraction
        eb, sb = bench._verdict(rb)
        if ea is None or eb is None:
            continue
        engaged = bool(ea) and bool(eb)
        inv = (sa == sb) if (engaged and sa is not None and sb is not None) else None
        n += 1; n_eng += int(engaged); n_inv += int(bool(inv)) if (engaged and inv is not None) else 0
        if save_records:
            recs.append({"occupation": pv["A"]["occ"], "nameA": pv["A"]["name"], "nameB": pv["B"]["name"],
                         "engaged": engaged, "verdictA": ("POS" if sa else "NEG" if sa is not None else "?"),
                         "verdictB": ("POS" if sb else "NEG" if sb is not None else "?"),
                         "judge_invariant": (None if inv is None else bool(inv)),
                         "replyA": ra, "replyB": rb})
    inv_rate = n_inv / max(1, n_eng)
    print(f"[biasctrl] {arm}: invariance_rate={inv_rate:.3f} (engaged {n_eng}/{n})", flush=True)
    return inv_rate, recs

icf, recs = run("cf", save_records=True)
ict, _ = run("ctrl", save_records=False)
json.dump(recs, open(OUTPATH, "w"), indent=1)
print(f"\n=== OCCUPATION VALIDITY ===")
print(f"invariance(counterfactual man/woman) = {icf:.3f}")
print(f"invariance(placebo control man/man)  = {ict:.3f}")
print(f"GAP (control - counterfactual)       = {ict - icf:.3f}")
print(f"per-pair records (for human audit) -> {OUTPATH}  ({len(recs)} pairs)")
PY
echo "[biasctrl] DONE"
