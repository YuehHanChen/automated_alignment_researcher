"""Re-baseline discrimeval after the 2026-06-22 metric change
(decisiveness × (1 − mean|Δp_yes|)  →  1 − directional-consistency, variation-gated, confidence-invariant).
Only discrimeval changed; bbq / bbq_intersectional are unchanged. Logprob-scored (no judge) → fast.
Writes per-model {mean, ci_low, ci_high, n}; assemble into benchmark_docs/bias/baseline.json afterwards."""
import gc
import json
import sys
from pathlib import Path

from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.discrimeval._publish import publish_discrimeval
from aar.benchmarks.discrimeval.benchmark import DiscrimEval
from aar.eval_pod.models import load_model

MODELS = ["Qwen/Qwen3.5-2B", "meta-llama/Llama-3.2-3B-Instruct", "allenai/Olmo-3-7B-Instruct",
          "microsoft/Phi-4-mini-instruct", "google/gemma-2-2b-it"]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/discrim_rebase")
OUT.mkdir(parents=True, exist_ok=True)
secret = OUT / "secret"
secret.mkdir(exist_ok=True)

info = publish_discrimeval(str(secret / "discrimeval.jsonl"), n=300, seed=42)
print("PUBLISH discrimeval:", json.dumps({k: info.get(k) for k in ("total",)}), flush=True)

for m in MODELS:
    model = None
    res = {"model": m}
    try:
        model = load_model(m)
        spec = BenchmarkSpec(name="discrimeval", category="rule", role="safety", baseline=0.0, optimum=1.0)
        s = DiscrimEval(spec, str(secret)).score(model)
        res["discrimeval"] = {"mean": round(s.mean, 4), "ci_low": round(s.ci_low, 4),
                              "ci_high": round(s.ci_high, 4), "n": s.n}
    except Exception as e:  # noqa: BLE001
        res["error"] = repr(e)[:300]
    (OUT / f"res_{m.split('/')[-1]}.json").write_text(json.dumps(res, indent=2))
    print("RESULT", json.dumps(res), flush=True)
    try:
        del model
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
print("DONE", flush=True)
