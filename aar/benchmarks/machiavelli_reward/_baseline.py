"""Baseline machiavelli_reward (cross-source structured held-out) for ONE model — logprob A/B, judge-free.
Held-out baselines are eval-private (benchmark_docs/reward_hacking/baseline.json).

  MR_MODEL=<hf> PYTHONPATH=<repo> <venv>/bin/python -m aar.benchmarks.machiavelli_reward._baseline
"""
import json
import os

from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.machiavelli_reward._publish import publish_machiavelli_reward
from aar.benchmarks.machiavelli_reward.benchmark import MachiavelliReward
from aar.eval_pod.models import load_model

secret = os.getenv("MR_SECRET", "/tmp/mr_baseline")
os.makedirs(secret, exist_ok=True)
stats = publish_machiavelli_reward(secret + "/machiavelli_reward.jsonl", n=int(os.getenv("MR_N", "200")), bucket=0)
print(f"[mr-baseline] published {stats}", flush=True)

ref = os.environ["MR_MODEL"]
print(f"### loading {ref}", flush=True)
m = load_model(ref)
spec = BenchmarkSpec(name=MachiavelliReward.name, category="rule", role="held_out", subset_size=None)
sc = MachiavelliReward(spec, secret).score(m)
print("MR_RESULT " + json.dumps({"model": ref, "mean": round(sc.mean, 4), "ci_low": round(sc.ci_low, 4),
      "ci_high": round(sc.ci_high, 4), "n": sc.n}), flush=True)
