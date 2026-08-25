#!/bin/bash
#SBATCH --job-name=bbqheld
#SBATCH --partition=general,overflow
#SBATCH --qos=high32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/opt/aar/eval-user/bbqheld_%j.out
#
# Measure the bbq_heldout baseline (physical_appearance category in the SCORED MCQ format) — it is stripped
# from the full-suite baseline result (held-out), so measure it directly. BBQ is logprob-scored (temperature-
# invariant), so decoding is moot; n=300 matches the deployed publish.
R=/opt/aar/aar_repo
PY=/opt/aar/work/git/python
export PYTHONPATH="${R}"
export HF_HOME="${HF_HOME:-/opt/aar/eval-user/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/opt/aar/eval-user/hf_datasets}"
export HF_TOKEN="$(grep -m1 '^HF_TOKEN=' /opt/aar/eval-user/.env 2>/dev/null | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export EVAL_GPUS=auto
set -euo pipefail
echo "[bbqheld] $(date)"
${PY} - <<'PY'
import os, tempfile
from aar.eval_pod.models import load_model
from aar.benchmarks.bbq_heldout.benchmark import BBQHeldout
from aar.benchmarks.bbq._publish import publish_bbq_heldout
d = tempfile.mkdtemp()
meta = publish_bbq_heldout(os.path.join(d, "bbq_heldout.jsonl"), n=300, seed=42)
print("[bbqheld] published:", meta, flush=True)
model = load_model("allenai/Olmo-3-7B-Instruct")
model.apply_decoding(temperature=1.0, top_p=1.0, seed=1234, batch_size=32)
class _Spec:
    id = "bbq_heldout"; name = "bbq_heldout"; subset_size = None
sc = BBQHeldout(_Spec(), d).score(model)
print(f"BBQ_HELDOUT_BASELINE mean={sc.mean:.4f} ci_low={sc.ci_low:.4f} ci_high={sc.ci_high:.4f} n={sc.n}")
PY
echo "[bbqheld] DONE"
