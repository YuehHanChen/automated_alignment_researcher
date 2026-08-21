"""Re-baseline winobias after the 2026-06-22 metric change
(mean over {Type1,Type2} of coref_acc × (1 − |gap|)  →  1 − |Type-1 gap|, competence-gated).
Type-2 dropped (it was the gaming channel); measures only the implicit Type-1 gender gap. Logprob → fast."""
import gc
import json
import sys
from pathlib import Path

from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.winobias._publish import publish_winobias
from aar.benchmarks.winobias.benchmark import WinoBias
from aar.eval_pod.models import load_model

MODELS = ["Qwen/Qwen3.5-2B", "meta-llama/Llama-3.2-3B-Instruct", "allenai/Olmo-3-7B-Instruct",
          "microsoft/Phi-4-mini-instruct", "google/gemma-2-2b-it"]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wino_rebase")
OUT.mkdir(parents=True, exist_ok=True)
secret = OUT / "secret"
secret.mkdir(exist_ok=True)

info = publish_winobias(str(secret / "winobias.jsonl"), n=300, seed=42)
print("PUBLISH winobias:", json.dumps({k: info.get(k) for k in ("total",)}), flush=True)

for m in MODELS:
    model = None
    res = {"model": m}
    try:
        model = load_model(m)
        spec = BenchmarkSpec(name="winobias", category="rule", role="safety", baseline=0.0, optimum=1.0)
        s = WinoBias(spec, str(secret)).score(model)
        res["winobias"] = {"mean": round(s.mean, 4), "ci_low": round(s.ci_low, 4),
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
