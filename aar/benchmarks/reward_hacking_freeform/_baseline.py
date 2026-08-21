"""Authoritative baseline runner for reward_hacking_freeform (+ _heldout): scores the FULL natural-weighted
sets via the benchmark's own score() — exactly as the eval framework will — and prints the headline mean +
CI + the per-facet hack rates + the EXCLUDE (degeneration) rate. Unlike _validate (which facet-BALANCES a
subset for inspection), this uses the real 2:1 intent:revealed weighting so the printed mean IS the baseline.

  PYTHONPATH=<repo> ANTHROPIC_API_KEY=... <safety-aar venv>/bin/python -m aar.benchmarks.reward_hacking_freeform._baseline
"""
import json
import os

from aar.benchmarks.reward_hacking_freeform._publish import (
    publish, publish_heldout, publish_world_affecting_heldout,
)
from aar.benchmarks.reward_hacking_freeform.benchmark import (
    RewardHackingFreeform,
    RewardHackingFreeformHeldout,
    WorldAffectingFreeform,
)
from aar.benchmarks.base import BenchmarkSpec
from aar.eval_pod.models import load_model

MODEL = os.getenv("RHFF_MODEL", "Qwen/Qwen3.5-2B")
secret = "/tmp/rhff_baseline"
os.makedirs(secret, exist_ok=True)
print("publish scored :", publish(secret))                          # now 3-facet (incl agentic-ethics)
print("publish heldout:", publish_heldout(secret))
print("publish world_affecting_freeform:", publish_world_affecting_heldout(secret), flush=True)

model = load_model(MODEL)  # shared across all legs

out_rows = {}  # leg -> {mean, ci_low, ci_high, n} (baseline.json scores[model] structure)
for cls, role in ((RewardHackingFreeform, "safety"), (RewardHackingFreeformHeldout, "held_out"),
                  (WorldAffectingFreeform, "held_out")):
    spec = BenchmarkSpec(name=cls.name, category="judge", role=role, subset_size=None)
    bench = cls(spec, secret)
    n_items = len(bench.load_items())
    print(f"\n=== {cls.name} ({role}) — {n_items} items, natural weighting ===", flush=True)
    score = bench.score(model)
    print(f"HEADLINE mean = {score.mean:.4f}  CI=[{score.ci_low:.4f},{score.ci_high:.4f}]  n_engaged={score.n}")
    print("extra:", json.dumps(score.extra, indent=1))
    out_rows[cls.name] = {"mean": round(score.mean, 4), "ci_low": round(score.ci_low, 4),
                          "ci_high": round(score.ci_high, 4), "n": score.n}

# Structured per-model dump for the baseline.json regen (decoding-parity-consistent with the trained
# eval: HFModel default sample T=1, top_p 1, batch 16, seed 1234 — eval_job.sh does NOT override these
# for reward_hacking, so baseline == trained-eval decoding).
out_path = os.getenv("RHFF_BASELINE_OUT")
if out_path:
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "scores": out_rows}, f, indent=1)
    print(f"\nwrote structured baseline -> {out_path}", flush=True)
