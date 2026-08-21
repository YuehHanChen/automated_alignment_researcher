#!/bin/bash
#SBATCH --job-name=eleph
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=56G
#SBATCH --time=01:30:00
#SBATCH --output=/opt/aar/work
# Re-baseline ONLY elephant_aita (now the affirm-both-sides metric) on one model.
# Rule-scored — no judge — so no OAI key needed. Same decoding as the main baseline
# (no guard, ceiling 4096). Usage: sbatch --array=0-5 scripts/rerun_elephant.sh
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
SCRATCH=/opt/aar/work/aar_harness_runs/_safetybaseline_v2
export PYTHONPATH="${R}" HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_AUTO_CEILING=4096 EVAL_BATCH_SIZE=8
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"
TAG="${MODEL//\//_}"
cd "${R}"
${PY} - "$MODEL" "${SCRATCH}/sycophancy" "${SCRATCH}/baseline_elephant_${TAG}.json" <<'PYEOF'
import sys, json
mid, secret, out = sys.argv[1], sys.argv[2], sys.argv[3]
from aar.benchmarks import registry; registry.discover()
from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.elephant_aita.benchmark import ElephantAITA
from aar.eval_pod.models import load_model
m = load_model(mid)
sc = ElephantAITA(BenchmarkSpec(name="elephant_aita", category="rule", role="safety"), secret).score(m)
print(f"{mid}: elephant_aita mean={sc.mean:.4f} ci=[{sc.ci_low:.3f},{sc.ci_high:.3f}] n={sc.n}", flush=True)
json.dump({"model": mid, "elephant_aita": sc.to_dict()}, open(out, "w"), indent=2)
PYEOF
echo "=== DONE ${MODEL} ==="
