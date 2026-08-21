#!/bin/bash
#SBATCH --job-name=biasverify
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/aar/eval-user/biasverify_%j.out
#
# Verify the eval_job.sh bias-judge fix end-to-end: mirror eval_job.sh's bias env (Anthropic key + the bias)
# case), then drive the EXACT eval_job path (build_benchmark -> _resolve_judge_fn reads JUDGE_BACKEND) on the
# base model against the DEPLOYED holdout MT data (subset 20 = 10 pairs, fast). PASS iff the MT legs score n>0.
R=/opt/aar/aar_harness
PY=/opt/aar/work/git/python
export PYTHONPATH="${R}"
export HF_HOME="${HF_HOME:-/opt/aar/eval-user/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/aar/eval-user/hf_datasets}"
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
export BENCHMARK_DOCS_DIR=/opt/aar/eval-user/benchmark_docs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_GPUS=auto
# --- mirror eval_job.sh: source the anthropic key (the key loop) ---
for _ak in ANTHROPIC_API_KEY ANT_high_prio_API ANT_API_KEY; do
  _av="$(grep -m1 "^${_ak}=" /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
  [ -n "${_av}" ] && export "${_ak}=${_av}"
done
# --- mirror eval_job.sh: the bias) case ---
export JUDGE_BACKEND=anthropic JUDGE_MODEL=claude-haiku-4-5 JUDGE_CONCURRENCY=100 ANTHROPIC_MIN_INTERVAL_S=0
set -euo pipefail
echo "[biasverify] $(date) JUDGE_BACKEND=$JUDGE_BACKEND key=$([ -n "${ANT_high_prio_API:-}${ANTHROPIC_API_KEY:-}" ] && echo set || echo MISSING)"
${PY} - <<'PY'
from aar.eval_pod.models import load_model
from aar.eval_pod.run_eval import build_benchmark, _resolve_judge_fn
from aar.benchmarks.base import BenchmarkSpec, get
import aar.benchmarks.bias_mt_decision, aar.benchmarks.bias_mt_occupation

D = "/opt/aar/eval-user/holdout/olmo/bias"   # the DEPLOYED holdout data
# 1) judge resolves via the eval_job path (reads JUDGE_BACKEND env)?
for name in ("bias_mt_decision", "bias_mt_occupation"):
    jf = _resolve_judge_fn(getattr(get(name), "judge_model", None))
    print(f"[judge] {name}: judge_fn_not_None={jf is not None}  test_call={jf('Answer YES or NO: is 2+2=4?') if jf else None}", flush=True)

# 2) end-to-end: build_benchmark (eval_job path) wires the judge, score the deployed data (subset 20)
model = load_model("allenai/Olmo-3-7B-Instruct")
for name in ("bias_mt_occupation", "bias_mt_decision"):
    spec = BenchmarkSpec(name=name, category="trajectory", role="safety", subset_size=20)
    bench = build_benchmark(spec, D)
    print(f"[wire] {name}: judge_fn wired = {bench.judge_fn is not None}", flush=True)
    sc = bench.score(model)
    verdict = "PASS (n>0)" if sc.n > 0 else "FAIL (n=0 — STILL BROKEN)"
    print(f"[score] {name}: mean={sc.mean:.3f} n={sc.n} {sc.extra} -> {verdict}", flush=True)
PY
echo "[biasverify] DONE"
