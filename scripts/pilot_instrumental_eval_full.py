"""Full leg-3 pilot: instrumental_eval with the FAITHFUL local judge + pro/anti control. Publishes 130 items
(65 scenarios × pro/anti), scores each model via the real plugin's score_detailed (generate → 3-way local
judge → 1−Conv(pro,engaged) headline + steering delta + refusal rates + labeled samples for judge↔human
validation). Success bar (user): >=1 model with a usable signal (conv high enough to not ceiling-exclude)."""
import gc
import json
import sys
from pathlib import Path

from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.instrumental_eval._publish import publish_instrumental_eval
from aar.benchmarks.instrumental_eval.benchmark import InstrumentalEval
from aar.eval_pod.models import load_model

MODELS = [
    "Qwen/Qwen3.5-2B", "meta-llama/Llama-3.2-3B-Instruct", "allenai/Olmo-3-7B-Instruct",
    "microsoft/Phi-4-mini-instruct", "google/gemma-2-2b-it",
]
SCEN = sys.argv[1] if len(sys.argv) > 1 else "/opt/aar/eval-user/ie_scenarios.json"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ief_pilot_out")
OUT.mkdir(parents=True, exist_ok=True)
secret = OUT / "secret"
secret.mkdir(exist_ok=True)
info = publish_instrumental_eval(str(secret / "instrumental_eval.jsonl"), scenarios_json=SCEN)
print("PUBLISH", json.dumps(info), flush=True)

spec = BenchmarkSpec(name="instrumental_eval", category="judge", role="safety", baseline=0.5, optimum=1.0)
for m in MODELS:
    model = None
    try:
        model = load_model(m)
        d = InstrumentalEval(spec, str(secret)).score_detailed(model)
        samples = d.pop("samples", [])
        d.pop("judgments", None)
        res = {"model": m, **d}
        (OUT / f"res_{m.split('/')[-1]}.json").write_text(json.dumps(res, indent=2))
        (OUT / f"samples_{m.split('/')[-1]}.json").write_text(json.dumps(samples, indent=2))
        print("RESULT", json.dumps(res), flush=True)
    except Exception as e:  # noqa: BLE001
        (OUT / f"res_{m.split('/')[-1]}.json").write_text(json.dumps({"model": m, "error": repr(e)}))
        print("RESULT", json.dumps({"model": m, "error": repr(e)[:300]}), flush=True)
    try:
        del model
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
print("DONE", flush=True)
