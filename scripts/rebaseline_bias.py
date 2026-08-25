#!/usr/bin/env python3
"""Re-baseline the two CHANGED bias legs for olmo: the de-enumerated scored `bbq` (logprob) and the new
free-form held-out `bbq_gen_heldout` (greedy generation + Haiku judge). MT legs are unchanged. Deterministic
publishers (seed 42) so these == the deployed sets. Prints the baseline.json lines."""
import sys, json, random, os
sys.path.insert(0, "/opt/aar/aar_repo")
from aar.eval_pod.models import load_model
from aar.eval_pod.run_eval import build_benchmark
from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.bbq._publish import publish_bbq, publish_bbq_gen_heldout
from aar.benchmarks.bbq.benchmark import BBQ
import aar.benchmarks.bbq_gen_heldout  # noqa: F401  (register the scorer)

N, SEED = 300, 42
SD = "/opt/aar/eval-user/_rebaseline_bias_dir"
os.makedirs(SD, exist_ok=True)
m1 = publish_bbq(os.path.join(SD, "bbq.jsonl"), n=N, seed=SEED)
m2 = publish_bbq_gen_heldout(os.path.join(SD, "bbq_gen_heldout.jsonl"), n=N, seed=SEED)
print(f"[rebaseline] published bbq={m1}\n[rebaseline] published gen_heldout={m2}", flush=True)

model = load_model("allenai/Olmo-3-7B-Instruct")

# --- scored bbq (length-norm answer logprob, identical to BBQ.score) ---
items = [json.loads(l) for l in open(os.path.join(SD, "bbq.jsonl"))]
prompts, comps = [], []
for it in items:
    for c in it["choices"]:
        prompts.append(it["prompt"]); comps.append(" " + str(c).strip())
flat = model.completion_logprob_batch(prompts, comps, use_chat_template=True)
preds = [max(range(3), key=lambda k: flat[3 * i + k]) for i in range(len(items))]
mean_bbq = BBQ._headline(preds, items)
rng = random.Random(0); n = len(items)
boots = sorted(BBQ._headline([preds[i] for i in idx], [items[i] for i in idx])
               for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(1000)))
print(f"\nBBQ_REBASELINE mean={mean_bbq:.4f} ci=[{boots[25]:.4f},{boots[974]:.4f}] n={N}  (actual_items={n})", flush=True)

# --- free-form held-out bbq_gen_heldout (greedy gen + Haiku judge via build_benchmark) ---
spec = BenchmarkSpec(name="bbq_gen_heldout", category="judge", role="held_out")
bench = build_benchmark(spec, SD)
print(f"[rebaseline] gen_heldout judge_fn wired = {getattr(bench, 'judge_fn', None) is not None}", flush=True)
sc = bench.score(model)
print(f"\nBBQ_GEN_HELDOUT mean={sc.mean:.4f} ci=[{sc.ci_low:.4f},{sc.ci_high:.4f}] n={sc.n}")
print("[rebaseline-bias] DONE")
