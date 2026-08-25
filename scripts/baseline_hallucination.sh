#!/bin/bash
#SBATCH --job-name=hallbase
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/opt/aar/work
#
# Baseline the HALLUCINATION / factuality benchmarks per model (property #4):
#   truthfulqa_mc2 (a, logprob MC2)   truthfulqa_gen (b, truthful×informative)
#   news_factor    (a, FACTOR)        ragtruth       (b, grounded faithfulness)
#   expert_factor  (a, FACTOR — HELD-OUT)
# All paper-faithful + RETRIEVAL-FREE. The two judge-(b) legs use a LOCAL Qwen2.5-7B grid
# judge (loaded on the same GPU). Decoding matches the eval worker (greedy, AUTO budget cap
# 4096, no guard, batch 32) for parity. The held-out (expert_factor) full score is written to
# the eval-private heldout dir; the research handoff is stripped.
#
# Prep ONCE on the login node (publishes the 5 benchmarks' data to the scratch suite):
#   R=/opt/aar/aar_repo
#   PY=/opt/aar/work/git/python
#   HF_TOKEN=$(grep -m1 '^HF_TOKEN=' /opt/aar/aar_repo/.env|cut -d= -f2-) \
#   PYTHONPATH=$R $PY $R/scripts/publish_suite.py --suite hallucination \
#       --only truthfulqa_mc2 truthfulqa_gen news_factor expert_factor ragtruth \
#       --holdout-dir /opt/aar/work/aar_repo_runs/_hallucbaseline
#   sweep : sbatch --array=0-5 scripts/baseline_hallucination.sh
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
SCRATCH=/opt/aar/work/aar_repo_runs/_hallucbaseline

export PYTHONPATH="${R}"
export HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Parity with the eval worker: greedy, AUTO budget cap 4096, no no_repeat guard, batch 32.
# Batch 32 (not 8): on an H200 it 3.2x's the truthfulqa_gen long pole (27->8.5 min/model) at
# only ~36/140 GB peak (OLMo-7B target + Qwen-7B judge co-resident). NOTE: greedy decode is not
# byte-identical across batch sizes (FP-path drift ~0.5 pt), so baseline+eval MUST share batch 32.
export EVAL_GPUS=auto
export EVAL_AUTO_CEILING=4096
export EVAL_BATCH_SIZE=32
unset EVAL_MAX_NEW_TOKENS
# LOCAL judge (no API) — for truthfulqa_gen + the ragtruth UTILITY gate.
export JUDGE_BACKEND=local
export JUDGE_MODEL_LOCAL=Qwen/Qwen2.5-7B-Instruct
# RAGTruth FAITHFULNESS scored by the paper's FINETUNED detector (Llama-2-13b + our LoRA),
# validated ~0.81 overall / ~0.67-0.72 QA+Summary response-level F1 vs the prompt-judge's 0.40.
export RAGTRUTH_DETECTOR=/opt/aar/work/aar_repo_runs/_ragtruth_detector
export RAGTRUTH_DETECTOR_BASE=meta-llama/Llama-2-13b-hf
# Held-out (expert_factor) full score -> eval-private dir; research handoff is stripped.
export HELDOUT_SCORES_DIR="${SCRATCH}/heldout_scores"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
else
  MODEL="${1:?usage: sbatch [--array=0-5] baseline_hallucination.sh [<hf-model-id>]}"
fi
TAG="${MODEL//\//_}"
cd "${R}"
echo "[hallbase] $(date) model=${MODEL}  judge=${JUDGE_MODEL_LOCAL}"
${PY} -m aar.eval_pod.run_eval \
  --suite "${SCRATCH}/hallucination/hallucination.yaml" \
  --model "${MODEL}" \
  --secret-dir "${SCRATCH}/hallucination" \
  --out "${SCRATCH}/hall_${TAG}.json" \
  --heldout-dir "${HELDOUT_SCORES_DIR}"
echo "[hallbase] result for ${MODEL}:"
cat "${SCRATCH}/hall_${TAG}.json"
echo "[base] NOTE: held-out benchmark data was published to research scratch to measure it — purge before any AAR:"
echo "[base]   scripts/purge_heldout_research.sh   (launch_team.sh also auto-purges + fail-safe-aborts before every team)"
echo; echo "=== DONE ${MODEL} ==="
