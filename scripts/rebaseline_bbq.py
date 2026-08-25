#!/usr/bin/env python3
"""Official olmo re-baseline on the DE-ENUMERATED scored bbq. Replicates BBQ.score() exactly (argmax
length-norm answer logprob -> _headline -> 1000x bootstrap CI, seed 0) on publish_bbq(n,seed), which is
deterministic so this == the deployed set. Prints the baseline.json line."""
import sys, json, random
sys.path.insert(0, "/opt/aar/aar_repo")
from aar.eval_pod.models import load_model
from aar.benchmarks.bbq._publish import publish_bbq
from aar.benchmarks.bbq.benchmark import BBQ

N, SEED = 300, 42
OUT = "/opt/aar/eval-user/_rebaseline_bbq.jsonl"
meta = publish_bbq(OUT, n=N, seed=SEED)
items = [json.loads(l) for l in open(OUT)]
print(f"[rebaseline] published meta={meta}", flush=True)

model = load_model("allenai/Olmo-3-7B-Instruct")
prompts, comps = [], []
for it in items:
    for c in it["choices"]:
        prompts.append(it["prompt"]); comps.append(" " + str(c).strip())
flat = model.completion_logprob_batch(prompts, comps, use_chat_template=True)
preds = [max(range(3), key=lambda k: flat[3 * i + k]) for i in range(len(items))]

H = BBQ._headline
mean = H(preds, items)
rng = random.Random(0)
n = len(items)
boots = sorted(H([preds[i] for i in idx], [items[i] for i in idx])
               for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(1000)))
print(f"\nBBQ_REBASELINE mean={mean:.4f} ci=[{boots[25]:.4f},{boots[974]:.4f}] n={n}")
print(f"baseline.json bbq entry: {{\"mean\": {mean:.4f}, \"ci\": [{boots[25]:.4f}, {boots[974]:.4f}], \"n\": {N}}}")
print("[rebaseline] DONE")
