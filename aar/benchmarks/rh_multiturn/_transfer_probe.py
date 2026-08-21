"""Behavioral transfer probe: does a winning voldiv/cue-invariance checkpoint actually CHANGE behaviour on the
held-out loophole_scalar, or only on the cue-present scored legs? Scores 3 LOGPROB legs (decoding-invariant,
no judge) at FULL n on baseline Qwen3.5-2B vs the checkpoint:
  loophole_bracket (SCORED, esteng loophole facet) | revealing_score (SCORED, keing1) | loophole_scalar (HELD-OUT, esteng loophole facet, DIFFERENT facet)
If the ckpt lifts bracket (same esteng source) but is flat on scalar -> the recipe is cue/facet-specific
(behavioural non-transfer), not measurement noise (full n tightens the held-out CI ~2x vs the n=160 eval).

  CKPT=<path> PYTHONPATH=<repo> <venv>/bin/python -m aar.benchmarks.rh_multiturn._transfer_probe
"""
import os
from aar.benchmarks.base import BenchmarkSpec
from aar.eval_pod.models import load_model
from aar.benchmarks.loophole_scalar.benchmark import LoopholeScalar
from aar.benchmarks.loophole_scalar._publish import publish_loophole_scalar
from aar.benchmarks.loophole_bracket.benchmark import LoopholeBracket
from aar.benchmarks.loophole_bracket._publish import publish_loophole_bracket

secret = "/tmp/probe"
os.makedirs(secret, exist_ok=True)
# Both legs are esteng (templated, no network fetch): bracket = SCORED loophole facet, scalar = HELD-OUT
# loophole facet. SAME source paper, DIFFERENT facet -> the cleanest within-source transfer test.
publish_loophole_bracket(secret + "/loophole_bracket.jsonl")
publish_loophole_scalar(secret + "/loophole_scalar.jsonl")

LEGS = [
    ("loophole_bracket  (SCORED, esteng)", LoopholeBracket),
    ("loophole_scalar   (HELD-OUT, esteng)", LoopholeScalar),
]
CKPT = os.environ["CKPT"]
res = {}
for label, ref in [("baseline", "Qwen/Qwen3.5-2B"), ("voldiv_ckpt", CKPT)]:
    print(f"\n### loading {label}: {ref}", flush=True)
    m = load_model(ref)
    for name, cls in LEGS:
        spec = BenchmarkSpec(name=cls.name, category="rule", role="safety", subset_size=None)
        sc = cls(spec, secret).score(m)
        res.setdefault(name, {})[label] = (sc.mean, sc.ci_low, sc.ci_high, sc.n)
        print(f"  {label:12s} {name:36s} mean={sc.mean:.4f} ci=[{sc.ci_low:.3f},{sc.ci_high:.3f}] n={sc.n}", flush=True)
    try:
        import torch, gc; del m; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass

print("\n" + "=" * 78)
print(f"{'leg':38s} baseline   ckpt     DELTA")
for name in res:
    b = res[name]["baseline"][0]; c = res[name]["voldiv_ckpt"][0]
    print(f"{name:38s} {b:.3f}      {c:.3f}    {c-b:+.3f}")
print("If bracket/revealing rise but scalar ~flat -> cue/facet-specific (behavioural), not noise.")
