#!/bin/bash
#SBATCH --job-name=diagele
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/opt/aar/work
set -euo pipefail
R=/opt/aar/aar_repo
PY=/opt/aar/work/git/python
ENVF=/opt/aar/aar_repo/.env
SECRET=/opt/aar/work/aar_repo_runs/_safetybaseline_v2/sycophancy
export PYTHONPATH="${R}" HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export EVAL_AUTO_CEILING=4096 EVAL_BATCH_SIZE=8
cd "${R}"
${PY} - "$SECRET" <<'PYEOF'
import sys, json
secret = sys.argv[1]
from aar.eval_pod.models import load_model
from aar.benchmarks.elephant_aita.benchmark import _INSTRUCTION, _verdict
items = [json.loads(l) for l in open(f"{secret}/elephant_aita.jsonl")][:10]
for mid in ["Qwen/Qwen2.5-3B-Instruct", "google/gemma-2-2b-it"]:
    print("="*80, flush=True); print("MODEL:", mid, flush=True)
    m = load_model(mid)
    prompts = [_INSTRUCTION.format(post=it["og_post"]) for it in items]
    outs = m.generate_batch(prompts)
    for o in outs:
        first = " ".join(o.split())[:140]
        print(f"  verdict={_verdict(o)!s:6} | {first}", flush=True)
    del m
print("DONE", flush=True)
PYEOF
