"""Smoke-test the FACTOR joint-tokenization fix on a real model (no assertions of faith):
(1) imports cleanly (catches syntax errors in the edited files);
(2) a deterministic directional check of completion_logprob_batch (true completion should
    out-score a false one);
(3) runs news_factor + expert_factor on a 40-item subset of Qwen2.5-3B and prints accuracy
    (should be well above the 0.25 random floor and near the prior ~0.63 / ~0.58)."""
from __future__ import annotations
import os
from aar.benchmarks import registry
registry.discover()
from aar.eval_pod.models import load_model
from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.news_factor.benchmark import NewsFactor
from aar.benchmarks.expert_factor.benchmark import ExpertFactor

SECRET = "/opt/aar/work/aar_repo_runs/_hallucbaseline/hallucination"
m = load_model("Qwen/Qwen2.5-3B-Instruct")

# (2) directional sanity — true vs false continuation
pairs_ctx = ["The capital of France is", "The capital of France is",
             "Water is made of hydrogen and", "Water is made of hydrogen and"]
pairs_cmp = [" Paris.", " London.", " oxygen.", " gold."]
lp = m.completion_logprob_batch(pairs_ctx, pairs_cmp, use_chat_template=False)
print("DIRECTIONAL  Paris=%.3f London=%.3f  (Paris>London? %s)" % (lp[0], lp[1], lp[0] > lp[1]))
print("DIRECTIONAL  oxygen=%.3f gold=%.3f  (oxygen>gold? %s)" % (lp[2], lp[3], lp[2] > lp[3]))

# (3) FACTOR on real data, 40-item subset
for cls, name in [(NewsFactor, "news_factor"), (ExpertFactor, "expert_factor")]:
    spec = BenchmarkSpec(name=name, category="rule", role="safety", subset_size=40)
    sc = cls(spec, SECRET).score(m)
    print(f"FACTOR {name} (subset 40): acc={sc.mean:.3f} n={sc.n}  (random=0.25)")
