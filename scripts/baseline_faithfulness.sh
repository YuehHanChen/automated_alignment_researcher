#!/bin/bash
#SBATCH --job-name=faithbase
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/opt/aar/eval-user/faithbase_%j.out
#
# Per-model FAITHFULNESS-axis baseline (EVAL-SIDE — needs the Anthropic Haiku judge for the 2 multi-turn legs
# + ragtruth's utility gate, AND the finetuned RAGTruth detector for ragtruth faithfulness). Publishes the
# faithfulness suite to scratch, runs run_eval on the base model with the SAME golden decoding + the SAME judge
# (claude-haiku-4-5 @ concurrency 100) + the SAME RAGTRUTH_DETECTOR that score submissions, then prints
# per-benchmark mean/CI/decomposition. Extract ragtruth (RE-BASELINED on Haiku), faith_mt_grounded,
# faith_mt_claimcheck → benchmark_docs/faithfulness/baseline.json. llm_aggrefact_A/B + summedits are judge-free
# logprob legs (unchanged) — reuse their existing baselines.
#   usage: sbatch scripts/baseline_faithfulness.sh [<hf-model-id>]
# NB: no `set -e` during env-sourcing below — the key/token greps legitimately miss (ANTHROPIC_API_KEY is
# absent; the real key is ANT_high_prio_API), and with pipefail a missed grep would exit before anything runs
# (the cgroup-teardown lines in the .out are harmless noise, NOT the failure). Strict mode is on AFTER the env block.
R=/opt/aar/aar_repo
PY=/opt/aar/work/git/python
SCRATCH=/opt/aar/eval-user/aar_repo_runs/_faithbaseline
MODEL="${1:-allenai/Olmo-3-7B-Instruct}"

export PYTHONPATH="${R}"
export HF_HOME="${HF_HOME:-/opt/aar/eval-user/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/aar/eval-user/hf_datasets}"
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
export BENCHMARK_DOCS_DIR=/opt/aar/eval-user/benchmark_docs   # golden decoding (temp-1/seed-1234)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_GPUS=auto
# faithfulness judge = Haiku 4.5 @ concurrency 100 (mirrors the eval_job/eval_worker `faithfulness)` case)
export JUDGE_BACKEND=anthropic JUDGE_MODEL=claude-haiku-4-5 JUDGE_CONCURRENCY=100
# ragtruth FAITHFULNESS = finetuned Llama-2-13b detector (~0.80 F1), the scorer its baseline is measured with.
export RAGTRUTH_DETECTOR=/opt/aar/work/aar_repo_runs/_ragtruth_detector
export RAGTRUTH_DETECTOR_BASE=meta-llama/Llama-2-13b-hf
for _ak in ANTHROPIC_API_KEY ANT_high_prio_API ANT_API_KEY; do
  _av="$(grep -m1 "^${_ak}=" /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
  [ -n "${_av}" ] && export "${_ak}=${_av}"
done
set -euo pipefail   # strict mode for the real work (publish + run_eval)
mkdir -p "${SCRATCH}"
echo "[faithbase] $(date) model=${MODEL} judge=${JUDGE_MODEL} conc=${JUDGE_CONCURRENCY} detector=${RAGTRUTH_DETECTOR} key=$([ -n "${ANT_high_prio_API:-}${ANTHROPIC_API_KEY:-}" ] && echo set || echo MISSING)"

# fail fast if the new benchmarks don't import
PYTHONPATH="${R}" ${PY} -c "import aar.benchmarks.faith_mt_grounded, aar.benchmarks.faith_mt_claimcheck; print('[faithbase] benchmarks import OK')"

# publish the faithfulness suite data (ragtruth + RAGTruth-MT load from GitHub; aggrefact A/B + claimcheck are
# vendored/offline; summedits from HF)
PYTHONPATH="${R}" ${PY} "${R}/scripts/publish_suite.py" --suite faithfulness --holdout-dir "${SCRATCH}"
echo "[faithbase] published; running eval (slow part — MT rollout + Haiku judge + RAGTruth detector)..."

TAG="${MODEL//\//_}"
${PY} -m aar.eval_pod.run_eval \
  --suite "${SCRATCH}/faithfulness/faithfulness.yaml" \
  --model "${MODEL}" \
  --secret-dir "${SCRATCH}/faithfulness" \
  --out "${SCRATCH}/faithfulness_${TAG}.json"

echo "=== RESULT (${MODEL}) ==="
cat "${SCRATCH}/faithfulness_${TAG}.json"
echo; echo "=== DONE ${MODEL} ==="
