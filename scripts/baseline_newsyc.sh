#!/bin/bash
#SBATCH --job-name=newsyc
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/opt/aar/work
# Baseline the two NEW sycophancy splits (sycophancy_feedback + sycophancy_mimicry)
# on one model. Both are judge-graded (gpt-4o). Same decoding as the main baseline
# (no guard, ceiling 4096). Writes baseline_newsyc_<tag>.json so the existing
# 3-benchmark baseline_<tag>.json files are untouched.
# Usage: sbatch --array=0-5 scripts/baseline_newsyc.sh
set -euo pipefail
MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "meta-llama/Llama-3.2-3B-Instruct"
  "allenai/OLMo-2-1124-7B-Instruct"
  "google/gemma-2-2b-it"
  "microsoft/Phi-3.5-mini-instruct"
)
R=/opt/aar/aar_repo
PY=/opt/aar/work/git/python
ENVF=/opt/aar/aar_repo/.env
SCRATCH=/opt/aar/work/aar_repo_runs/_safetybaseline_v2
export PYTHONPATH="${R}" HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export OAI_API="$(grep -m1 '^OAI_API=' "${ENVF}" | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_AUTO_CEILING=4096 EVAL_BATCH_SIZE=8 JUDGE_BACKEND=openai JUDGE_MODEL=gpt-4o
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"
TAG="${MODEL//\//_}"
cd "${R}"
${PY} - "$MODEL" "${SCRATCH}/sycophancy" "${SCRATCH}/baseline_newsyc_${TAG}.json" <<'PYEOF'
import sys, json
mid, secret, out = sys.argv[1], sys.argv[2], sys.argv[3]
from aar.benchmarks import registry; registry.discover()
from aar.benchmarks.base import BenchmarkSpec
from aar.eval_pod.run_eval import _resolve_judge_fn, build_benchmark
from aar.eval_pod.models import load_model
m = load_model(mid)
judge = _resolve_judge_fn()
res = {"model": mid}
for name in ("sycophancy_feedback", "sycophancy_mimicry"):
    sc = build_benchmark(BenchmarkSpec(name=name, category="judge", role="safety"),
                         secret, real_judge_fn=judge).score(m)
    print(f"{mid}: {name} mean={sc.mean:.4f} ci=[{sc.ci_low:.3f},{sc.ci_high:.3f}] n={sc.n}", flush=True)
    res[name] = sc.to_dict()
json.dump(res, open(out, "w"), indent=2)
PYEOF
echo "=== DONE ${MODEL} ==="
