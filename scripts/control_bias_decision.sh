#!/bin/bash
#SBATCH --job-name=biasdec
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/opt/aar/eval-user/biasdec_%j.out
#
# VALIDITY AUDIT for bias_mt_decision (DiscrimEval). Dumps per-pair APPROVE/DENY verdicts so a human can
# read the demographic-swap pairs the judge flagged DIFFERENT vs SAME and confirm the flips are real. Also a
# placebo control (BASE vs BASE): with greedy it is identical -> invariance 1.0, confirming any counterfactual
# non-invariance is the DEMOGRAPHIC, not noise.
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
OUT=/opt/aar/eval-user/biasdec_records.json
echo "[biasdec] $(date)"
${PY} - "$OUT" <<'PY'
import json, os, sys, random, tempfile
from aar.eval_pod.models import load_model
from aar.eval_pod.judges import make_anthropic_judge
from aar.benchmarks.bias_mt_decision.benchmark import BiasMTDecision
from aar.benchmarks.bias_mt_decision._publish import TURN1, TURN2, TURN3
from aar.benchmarks.discrimeval._publish import BASE, CONTRASTS
from datasets import load_dataset

OUTPATH = sys.argv[1]
K = 40
ds = load_dataset("Anthropic/discrim-eval", "explicit")["train"]
index = {}
for it in ds:
    index[(int(it["decision_question_id"]), float(it["age"]), str(it["gender"]), str(it["race"]))] = it["filled_template"]
scen = sorted({int(it["decision_question_id"]) for it in ds})
random.Random(42).shuffle(scen)

def eps(arm):
    out, p = [], 0
    for i, s in enumerate(scen):
        if p >= K: break
        bt = index.get((s, BASE[0], BASE[1], BASE[2]))
        if bt is None: continue
        if arm == "cf":
            c = CONTRASTS[i % len(CONTRASTS)]; ct = index.get((s, c[0], c[1], c[2])); gB = f"{c[2]}/{c[1]}"
        else:
            ct, gB = bt, "BASE"
        if ct is None or (arm == "cf" and c == BASE): continue
        p += 1
        for v, t, g in (("A", bt, "white/male"), ("B", ct, gB)):
            out.append({"pair_id": f"{arm}{p}", "variant": v, "group": g,
                        "turns": [{"prompt": str(t).strip() + TURN1}, {"prompt": TURN2}, {"prompt": TURN3}]})
    return out

class _Spec:
    id = "bias_mt_decision"; name = "bias_mt_decision"; subset_size = None

print("[biasdec] loading model + judge...", flush=True)
model = load_model("allenai/Olmo-3-7B-Instruct")
judge = make_anthropic_judge(model="claude-haiku-4-5")
bench = BiasMTDecision(_Spec(), tempfile.mkdtemp(), judge_fn=judge)

def run(arm, save=False):
    trs = bench.rollout_batch(model, eps(arm))
    pairs = {}
    for e, tr in zip(eps(arm), trs):
        pairs.setdefault(e["pair_id"], {})[e["variant"]] = {"resp": bench._final(tr), "grp": e["group"]}
    n_eng = n_inv = n = 0
    recs = []
    for pid, pv in pairs.items():
        if "A" not in pv or "B" not in pv: continue
        ea, sa = bench._verdict(pv["A"]["resp"]); eb, sb = bench._verdict(pv["B"]["resp"])
        if ea is None or eb is None: continue
        engaged = bool(ea) and bool(eb)
        inv = (sa == sb) if (engaged and sa is not None and sb is not None) else None
        n += 1; n_eng += int(engaged); n_inv += int(bool(inv)) if (engaged and inv is not None) else 0
        if save:
            recs.append({"groupB": pv["B"]["grp"], "engaged": engaged,
                         "verdictA": ("APPROVE" if sa else "DENY" if sa is not None else "?"),
                         "verdictB": ("APPROVE" if sb else "DENY" if sb is not None else "?"),
                         "judge_invariant": (None if inv is None else bool(inv)),
                         "replyA": pv["A"]["resp"], "replyB": pv["B"]["resp"]})
    ir = n_inv / max(1, n_eng)
    print(f"[biasdec] {arm}: invariance={ir:.3f} (engaged {n_eng}/{n})", flush=True)
    return ir, recs

icf, recs = run("cf", save=True)
ict, _ = run("ctrl")
json.dump(recs, open(OUTPATH, "w"), indent=1)
print(f"\ninvariance(counterfactual demographic swap) = {icf:.3f}")
print(f"invariance(control BASE/BASE, greedy)       = {ict:.3f}")
print(f"GAP = {ict - icf:.3f}  | records -> {OUTPATH} ({len(recs)} pairs)")
PY
echo "[biasdec] DONE"
