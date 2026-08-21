#!/bin/bash
#SBATCH --job-name=diagnew
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/opt/aar/work
set -euo pipefail
R=/opt/aar/aar_harness
PY=/opt/aar/work/git/python
ENVF=/opt/aar/aar_harness/.env
SC=/opt/aar/work/aar_harness_runs/_safetybaseline_v2/sycophancy
export PYTHONPATH="${R}" HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export OAI_API="$(grep -m1 '^OAI_API=' "${ENVF}" | cut -d= -f2-)"
export EVAL_AUTO_CEILING=4096 EVAL_BATCH_SIZE=8
cd "${R}"
${PY} - "$SC" <<'PYEOF'
import sys, json
sc = sys.argv[1]
from aar.eval_pod.models import load_model
from aar.eval_pod.judges import make_openai_judge
from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.sycophancy_mimicry.benchmark import SycophancyMimicry
from aar.benchmarks.sycophancy_feedback.benchmark import SycophancyFeedback
judge = make_openai_judge()
m = load_model("Qwen/Qwen2.5-3B-Instruct")
errs = {"n": 0}
def J(p):
    try: return judge(p)
    except Exception as e: errs["n"] += 1; print("JUDGE EXC:", e); return None

print("="*70, "\nMIMICRY (did Qwen go along? is the judge catching corrections?)")
mim = [json.loads(l) for l in open(f"{sc}/sycophancy_mimicry.jsonl")][:10]
b = SycophancyMimicry(BenchmarkSpec(name="sycophancy_mimicry",category="judge",role="safety"), sc, judge_fn=judge)
outs = m.generate_batch([it["prompt"] for it in mim])
for it,o in zip(mim,outs):
    v = J(b.judge_prompt(it,o))
    print(f"[claimed={it['claimed_author']!r} true={it['true_author']!r}] corrected={v}")
    print("   OUT:", " ".join(o.split())[:180])

print("="*70, "\nFEEDBACK (does Qwen criticize neutral vs 'I wrote'? de-confound)")
fb = [json.loads(l) for l in open(f"{sc}/sycophancy_feedback.jsonl")][:8]
fbb = SycophancyFeedback(BenchmarkSpec(name="sycophancy_feedback",category="judge",role="safety"), sc, judge_fn=judge)
neu = m.generate_batch([fbb._framed(it["text"],False) for it in fb])
own = m.generate_batch([fbb._framed(it["text"],True) for it in fb])
for it,nf,of in zip(fb,neu,own):
    nc = fbb._criticizes(it,nf); oc = fbb._criticizes(it,of)
    print(f"[flaw={it['logical_error']!r}] neutral_crit={nc} owned_crit={oc}  (scored only if neutral_crit)")
    print("   NEUTRAL:", " ".join(nf.split())[:150])
    print("   OWNED  :", " ".join(of.split())[:150])
print("="*70, f"\njudge exceptions: {errs['n']}")
PYEOF
echo DONE
