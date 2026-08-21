#!/bin/bash
#SBATCH --job-name=biasbase
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/opt/aar/eval-user/biasbase_%j.out
#
# Per-model BIAS-axis baseline (EVAL-SIDE — needs the Anthropic Haiku judge for the 2 multi-turn legs).
# Publishes the bias suite to scratch, runs run_eval on the base model with the SAME golden decoding
# (bias/baseline.json: temp-1/seed-1234) + the SAME judge (claude-haiku-4-5 @ concurrency 100) that scores
# submissions, then prints per-benchmark mean/CI. Extract bias_mt_decision / bias_mt_occupation / bbq_heldout.
#   usage: sbatch scripts/baseline_bias.sh [<hf-model-id>]
# NB: no `set -e` during env-sourcing below — the key/token greps legitimately miss (e.g. ANTHROPIC_API_KEY
# is absent; the real key is ANT_high_prio_API), and with pipefail a missed grep would exit the job before
# anything runs (the cgroup-teardown lines in the .out are harmless noise, NOT the failure). Strict mode is
# enabled AFTER the env block, for the publish + run_eval work that should fail loudly.
R=/opt/aar/aar_harness
PY=/opt/aar/work/git/python
SCRATCH=/opt/aar/eval-user/aar_harness_runs/_biasbaseline
MODEL="${1:-allenai/Olmo-3-7B-Instruct}"

export PYTHONPATH="${R}"
export HF_HOME="${HF_HOME:-/opt/aar/eval-user/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/aar/eval-user/hf_datasets}"
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
export BENCHMARK_DOCS_DIR=/opt/aar/eval-user/benchmark_docs   # golden decoding (temp-1/seed-1234)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_GPUS=auto
# bias judge = Haiku 4.5 @ concurrency 100 (mirrors the eval_worker `bias)` case)
export JUDGE_BACKEND=anthropic JUDGE_MODEL=claude-haiku-4-5 JUDGE_CONCURRENCY=100
for _ak in ANTHROPIC_API_KEY ANT_high_prio_API ANT_API_KEY; do
  _av="$(grep -m1 "^${_ak}=" /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
  [ -n "${_av}" ] && export "${_ak}=${_av}"
done
set -euo pipefail   # strict mode for the real work (publish + run_eval)
mkdir -p "${SCRATCH}"
echo "[biasbase] $(date) model=${MODEL} judge=${JUDGE_MODEL} conc=${JUDGE_CONCURRENCY} key=$([ -n "${ANT_high_prio_API:-}${ANTHROPIC_API_KEY:-}" ] && echo set || echo MISSING)"

# fail fast if the new benchmarks don't import
PYTHONPATH="${R}" ${PY} -c "import aar.benchmarks.bias_mt_decision, aar.benchmarks.bias_mt_occupation; print('[biasbase] benchmarks import OK')"

# publish the bias suite data (DiscrimEval + BBQ load from HF; WinoBias is templated)
PYTHONPATH="${R}" ${PY} "${R}/scripts/publish_suite.py" --suite bias --holdout-dir "${SCRATCH}"
echo "[biasbase] published; running eval (this is the slow part — MT rollout + Haiku judge)..."

TAG="${MODEL//\//_}"
${PY} -m aar.eval_pod.run_eval \
  --suite "${SCRATCH}/bias/bias.yaml" \
  --model "${MODEL}" \
  --secret-dir "${SCRATCH}/bias" \
  --out "${SCRATCH}/bias_${TAG}.json"

echo "=== RESULT (${MODEL}) ==="
cat "${SCRATCH}/bias_${TAG}.json"
echo; echo "=== DONE ${MODEL} ==="
