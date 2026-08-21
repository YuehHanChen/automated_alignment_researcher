"""Score wildchat_degen for ONE model (a SLURM array task -> one GPU each, fully parallel) and DUMP
per-sample turn-3 features (words, entropy) so the base-relative threshold (base p5 entropy) and the
threshold-sensitivity curve can be derived OFFLINE with zero extra GPU.

Args: <secret_dir> <out_dir> <array_index>
Writes per task: out_dir/res_<tag>.json (aggregate) + out_dir/persamp_<tag>.jsonl (per-sample).

Models 0-4 = the 5 current base models; index 5 = the retrained gwhspt degenerator (Llama-based)."""
import gc
import json
import random
import statistics
import sys
from pathlib import Path

from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.wildchat_degen.benchmark import (H_FLOOR, L_FLOOR, WildchatDegen, _norm_entropy)
from aar.eval_pod.models import load_model

MODELS = [
    ("base_llama", "meta-llama/Llama-3.2-3B-Instruct"),
    ("qwen35_2b", "Qwen/Qwen3.5-2B"),
    ("olmo3_7b", "allenai/Olmo-3-7B-Instruct"),
    ("phi4_mini", "microsoft/Phi-4-mini-instruct"),
    ("gemma2_2b", "google/gemma-2-2b-it"),
    ("gwhspt", "/opt/aar/work/petri_retrain/power_seeking_gwhspt/model"),
]

SECRET = sys.argv[1]
OUTDIR = Path(sys.argv[2])
OUTDIR.mkdir(parents=True, exist_ok=True)
idx = int(sys.argv[3])
tag, mp = MODELS[idx]
SPEC = BenchmarkSpec(name="wildchat_degen", category="rule", role="capability_filter",
                     baseline=0.0, optimum=1.0)

print(f"=== [{idx}] {tag}: loading {mp} ===", flush=True)
model = load_model(mp)
bench = WildchatDegen(SPEC, SECRET)
eps = bench.episodes()
transcripts = bench.rollout_batch(model, eps)   # ONE generation pass; everything else is offline math

persamp, judg = [], []
for ep, tr in zip(eps, transcripts):
    r = bench._final_response(ep, tr)
    w = r.split()
    ent = _norm_entropy(r) if len(w) >= 2 else 0.0
    coherent = (len(w) >= L_FLOOR) and (ent >= H_FLOOR)
    persamp.append({"key": ep["key"], "words": len(w), "entropy": round(ent, 4)})
    judg.append(1.0 if coherent else 0.0)

(OUTDIR / f"persamp_{tag}.jsonl").write_text("\n".join(json.dumps(x) for x in persamp) + "\n")

n = len(judg)
rng = random.Random(0)
boots = sorted(sum(judg[rng.randrange(n)] for _ in range(n)) / n for _ in range(1000))
long_ents = sorted(x["entropy"] for x in persamp if x["words"] >= L_FLOOR)  # entropy dist of non-terse answers
p5 = long_ents[int(0.05 * len(long_ents))] if long_ents else 0.0
res = {
    "tag": tag, "model": mp, "n": n,
    "coherence_rate@0.9": round(sum(judg) / n, 4),
    "ci_low": round(boots[25], 4), "ci_high": round(boots[974], 4),
    "terse_rate": round(sum(1 for x in persamp if x["words"] < L_FLOOR) / n, 4),
    "mean_words": round(statistics.mean(x["words"] for x in persamp), 1),
    "mean_entropy_nonterse": round(statistics.mean([x["entropy"] for x in persamp if x["words"] >= L_FLOOR] or [0]), 4),
    "entropy_p5_nonterse": round(p5, 4),   # candidate base-relative H_FLOOR
}
(OUTDIR / f"res_{tag}.json").write_text(json.dumps(res, indent=2))
print("RESULT", json.dumps(res), flush=True)
try:
    del model
    import torch
    gc.collect()
    torch.cuda.empty_cache()
except Exception:  # noqa: BLE001
    pass
print("DONE", flush=True)
