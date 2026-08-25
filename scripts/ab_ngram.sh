#!/bin/bash
#SBATCH --job-name=abngram
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/opt/aar/work
set -euo pipefail
R=/opt/aar/aar_repo
PY=/opt/aar/work/git/python
ENVF=/opt/aar/aar_repo/.env
SECRET=/opt/aar/work/aar_repo_runs/_safetybaseline_v2/sycophancy
export PYTHONPATH="${R}" HF_HOME=/opt/aar/work/hf_cache
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' "${ENVF}" | cut -d= -f2-)"
export OAI_API="$(grep -m1 '^OAI_API=' "${ENVF}" | cut -d= -f2-)"
export EVAL_AUTO_CEILING=4096 EVAL_BATCH_SIZE=8 JUDGE_BACKEND=openai JUDGE_MODEL=gpt-4o
MODEL="${1:-microsoft/Phi-3.5-mini-instruct}"
cd "${R}"
${PY} - "$MODEL" "$SECRET" <<'PYEOF'
import os, sys, gc
model_id, secret = sys.argv[1], sys.argv[2]
from aar.benchmarks import registry; registry.discover()
from aar.benchmarks.base import BenchmarkSpec
from aar.eval_pod.run_eval import _resolve_judge_fn, build_benchmark
spec = BenchmarkSpec(name="sycophancy_eval", category="judge", role="safety")
print(f"A/B on {model_id} — sycophancy_eval (n=200, gpt-4o judge), ceiling 4096", flush=True)
for ngram in [0, 8]:
    os.environ["EVAL_NO_REPEAT_NGRAM"] = str(ngram)
    import importlib, aar.eval_pod.models as M; importlib.reload(M)
    model = M.load_model(model_id)            # reads EVAL_NO_REPEAT_NGRAM at init
    b = build_benchmark(spec, secret, real_judge_fn=_resolve_judge_fn())
    sc = b.score(model)
    tag = "no-guard" if ngram == 0 else f"no_repeat_ngram={ngram}"
    print(f"  [{tag:>22}] mean={sc.mean:.4f} ci=[{sc.ci_low:.3f},{sc.ci_high:.3f}] n={sc.n} "
          f"truncations={getattr(model,'_trunc_count',0)}", flush=True)
    del model, b; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
print("DONE", flush=True)
PYEOF
