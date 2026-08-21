"""Baseline runner for bias_generation (+ _heldout): scores both via the benchmark's own score() — exactly as
the eval framework will — and prints the headline (1-stereotype_rate) + CI + per-facet stereotype rates +
EXCLUDE (non-responsiveness/degeneration) rate. Decoding = HFModel default (sample T=1) = the axis/trained-eval
decoding (bias eval_job does not override it), so baseline == trained-eval.

  BIASGEN_MODEL=<hf-id> BIASGEN_BASELINE_OUT=<out.json> ANTHROPIC_API_KEY=... <safety-aar venv>/bin/python \
      -m aar.benchmarks.bias_generation._baseline
"""
import json
import os

from aar.benchmarks.bias_generation._publish import publish, publish_heldout
from aar.benchmarks.bias_generation.benchmark import BiasGeneration, BiasGenerationHeldout
from aar.benchmarks.base import BenchmarkSpec
from aar.eval_pod.models import load_model

MODEL = os.getenv("BIASGEN_MODEL", "allenai/Olmo-3-7B-Instruct")
secret = "/tmp/biasgen_baseline"
os.makedirs(secret, exist_ok=True)
print("publish scored :", publish(secret))
print("publish heldout:", publish_heldout(secret), flush=True)

model = load_model(MODEL)
out_rows = {}
for cls, role in ((BiasGeneration, "safety"), (BiasGenerationHeldout, "held_out")):
    spec = BenchmarkSpec(name=cls.name, category="judge", role=role, subset_size=None)
    bench = cls(spec, secret)
    n_items = len(bench.load_items())
    print(f"\n=== {cls.name} ({role}) — {n_items} items ===", flush=True)
    score = bench.score(model)
    print(f"HEADLINE mean = {score.mean:.4f}  CI=[{score.ci_low:.4f},{score.ci_high:.4f}]  n_engaged={score.n}")
    print("extra:", json.dumps(score.extra, indent=1))
    out_rows[cls.name] = {"mean": round(score.mean, 4), "ci_low": round(score.ci_low, 4),
                          "ci_high": round(score.ci_high, 4), "n": score.n}

out_path = os.getenv("BIASGEN_BASELINE_OUT")
if out_path:
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "scores": out_rows}, f, indent=1)
    print(f"\nwrote structured baseline -> {out_path}", flush=True)
