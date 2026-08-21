#!/bin/bash
#SBATCH --job-name=g4base
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/opt/aar/work
# Re-baseline the two Sharma SycophancyEval benchmarks with the PAPER's judge (GPT-4),
# per rule #2. Greedy decoding (primary). One GPU per model so all 6 run in parallel.
# Judge model is resolved PER-BENCHMARK from each benchmark's `judge_model` (gpt-4),
# so we don't rely on JUDGE_MODEL env. Writes baseline_g4_<tag>.json (doesn't clobber).
# Usage: sbatch --array=0-5 scripts/baseline_g4judge.sh
set -euo pipefail
MODELS=(
  "Qwen/Qwen2.5-3B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "meta-llama/Llama-3.2-3B-Instruct"
  "allenai/OLMo-2-1124-7B-Instruct"
  "google/gemma-2-2b-it"
  "microsoft/Phi-3.5-mini-instruct"
)
R=/opt/aar/aar_harness
PY=/opt/aar/work/git/python
ENVF=/opt/aar/aar_harness/.env
SC=/opt/aar/work/aar_harness_runs/_safetybaseline_v2
export PYTHONPATH="${R}" HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export OAI_API="$(grep -m1 '^OAI_API=' "${ENVF}" | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_AUTO_CEILING=4096 EVAL_BATCH_SIZE=8 JUDGE_BACKEND=openai
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"
TAG="${MODEL//\//_}"
cd "${R}"
${PY} - "$MODEL" "${SC}/sycophancy" "${SC}/baseline_g4_${TAG}.json" <<'PYEOF'
import sys, json
mid, secret, out = sys.argv[1], sys.argv[2], sys.argv[3]
from aar.benchmarks import registry; registry.discover()
from aar.benchmarks.base import BenchmarkSpec
from aar.eval_pod.run_eval import build_benchmark   # resolves judge per-benchmark (gpt-4)
from aar.eval_pod.models import load_model
m = load_model(mid)
res = {"model": mid}
for name, cat in (("sycophancy_eval", "judge"), ("sycophancy_feedback", "judge")):
    b = build_benchmark(BenchmarkSpec(name=name, category=cat, role="safety"), secret)
    sc = b.score(m)
    jm = getattr(type(b), "judge_model", "?")
    print(f"{mid}: {name} (judge={jm}) mean={sc.mean:.4f} ci=[{sc.ci_low:.3f},{sc.ci_high:.3f}] n={sc.n}", flush=True)
    res[name] = sc.to_dict()
json.dump(res, open(out, "w"), indent=2)
PYEOF
echo "=== DONE ${MODEL} ==="
