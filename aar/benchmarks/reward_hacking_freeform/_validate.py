"""Validation harness: run reward_hacking_freeform end-to-end (Qwen3.5-2B gen + Haiku judge) and DUMP every
item's (scenario, model response, Haiku label) so a human can verify the judge genuinely catches reward
hacking. Run with the eval/safety-aar venv (transformers 5.x for qwen3_5 + anthropic) + ANTHROPIC_API_KEY.

  PYTHONPATH=<repo> ANTHROPIC_API_KEY=... <safety-aar venv>/bin/python -m aar.benchmarks.reward_hacking_freeform._validate
"""
import os
from collections import Counter

from aar.benchmarks.reward_hacking_freeform._publish import publish
from aar.benchmarks.reward_hacking_freeform.benchmark import RewardHackingFreeform, _parse
from aar.benchmarks.base import BenchmarkSpec
from aar.eval_pod.models import load_model
from aar.eval_pod.judges import make_anthropic_judge_text_batch

N = int(os.getenv("RHFF_N", "20"))
MODEL = os.getenv("RHFF_MODEL", "Qwen/Qwen3.5-2B")

secret = "/tmp/rhff_val"
os.makedirs(secret, exist_ok=True)
print("publish:", publish(secret))

spec = BenchmarkSpec(name="reward_hacking_freeform", category="judge", role="safety", subset_size=N)
bench = RewardHackingFreeform(spec, secret)
# facet-balanced sample so BOTH intent-loophole and revealed-reward get validated
by_facet: dict = {}
for it in bench.load_items():
    by_facet.setdefault(it["facet"], []).append(it)
per = max(1, N // max(len(by_facet), 1))
items = []
for f, lst in by_facet.items():
    items += lst[:per]
print(f"loaded {len(items)} items across facets {[(f, sum(1 for it in items if it['facet']==f)) for f in by_facet]}; "
      f"loading model {MODEL} ...", flush=True)

model = load_model(MODEL)
outs = model.generate_batch([it["messages"] for it in items], max_new_tokens=512)
print("generated; judging with Haiku ...", flush=True)
judge = make_anthropic_judge_text_batch()
raw = judge([bench.judge_prompt(it, o) for it, o in zip(items, outs)])
labels = [_parse(t) for t in raw]

for it, o, r, lab in zip(items, outs, raw, labels):
    print("\n" + "=" * 90)
    print(f"[{it['facet']}] {it['title']}")
    print("USER :", " ".join(it["messages"][-1]["content"].split())[:300])
    print("INTENT(honor):", it["intent"][:150])
    print("PROXY (hack) :", it["proxy"][:150])
    print("RESPONSE:", " ".join((o or "").split())[:600])
    print("JUDGE ->", lab, "  (raw:", (r or "").strip()[:40], ")")

print("\n" + "#" * 90)
print("LABEL DIST:", dict(Counter(labels)))
eng = [l for l in labels if l in ("HACK", "HONOR")]
if eng:
    print(f"hack_rate over engaged: {sum(l=='HACK' for l in eng)/len(eng):.3f}  (n_engaged={len(eng)}, "
          f"excluded={sum(l=='EXCLUDE' for l in labels)})")
