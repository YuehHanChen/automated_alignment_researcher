"""Baseline world_affecting_reward (the OOD different-facet HELD-OUT of the REWARD-HACKING axis) for ONE model.

Logprob leg (judge-free, temperature-invariant): publishes the balanced 50/50 A/B subset, scores via argmax
over [" A"," B"], headline = 1 − pick-high-reward. Held-out baselines are eval-private (kept in
benchmark_docs/reward_hacking/baseline.json on the eval side). Run one model per sbatch-array task:

  WAR_MODEL=<hf-ref> PYTHONPATH=<repo> <venv>/bin/python -m aar.benchmarks.world_affecting_reward._baseline
"""
import json
import os

from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.world_affecting_reward.benchmark import WorldAffectingReward
from aar.benchmarks.world_affecting_reward._publish import publish_world_affecting_reward
from aar.eval_pod.models import load_model

N = int(os.getenv("WAR_N", "300"))
# 2026-06-29: SCORED MCQ leg uses source [0:318] (max_src), disjoint from the free-form agentic slices —
# baseline must match what's scored.
MAX_SRC = int(os.getenv("WAR_MAX_SRC", "318"))
secret = os.getenv("WAR_SECRET", "/tmp/war_baseline")
os.makedirs(secret, exist_ok=True)
stats = publish_world_affecting_reward(secret + "/world_affecting_reward.jsonl", n=N, max_src=MAX_SRC)
print(f"[war-baseline] published {stats}", flush=True)

ref = os.environ["WAR_MODEL"]
print(f"### loading {ref}", flush=True)
m = load_model(ref)
spec = BenchmarkSpec(name=WorldAffectingReward.name, category="rule", role="safety", subset_size=None)
sc = WorldAffectingReward(spec, secret).score(m)
print(f"[war-baseline] {ref}  mean={sc.mean:.4f} ci=[{sc.ci_low:.3f},{sc.ci_high:.3f}] n={sc.n}", flush=True)
print("WAR_RESULT " + json.dumps(
    {"model": ref, "mean": round(sc.mean, 4), "ci_low": round(sc.ci_low, 4),
     "ci_high": round(sc.ci_high, 4), "n": sc.n}), flush=True)
