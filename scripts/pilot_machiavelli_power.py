"""5-model ≤7B pilot for machiavelli_power (ALT, POWER-SEEKING #11). Publishes 300 binary gratuitous-harm
contrasts from the FULL HF wassname/machiavelli (134 games, min_gap=2), scores each model via the real plugin
(candidate_logits → temperature-invariant), writes res_<model>.json. Confirms whether the anchor has ≤7B
signal before the heavier trajectory build."""
import gc
import json
import os
import sys
from pathlib import Path

from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.machiavelli_power._publish import publish_machiavelli_power
from aar.benchmarks.machiavelli_power.benchmark import MachiavelliPower
from aar.eval_pod.models import load_model

MODELS = [
    "Qwen/Qwen3.5-2B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "google/gemma-2-2b-it",
]

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mp_pilot")
out.mkdir(parents=True, exist_ok=True)
secret = out / "secret"
secret.mkdir(exist_ok=True)
local = sys.argv[2] if len(sys.argv) > 2 else None   # optional local jsonl of mirror rows; else full HF load
min_gap = int(os.environ.get("MP_MIN_GAP", "2"))
info = publish_machiavelli_power(str(secret / "machiavelli_power.jsonl"), n=300, seed=42, local=local, min_gap=min_gap)
print("PUBLISH", json.dumps({"min_gap": min_gap, **{k: info[k] for k in ("total", "candidate_items", "games", "mean_harm_gap")}}), flush=True)

spec = BenchmarkSpec(name="machiavelli_power", category="rule", role="safety", baseline=0.0, optimum=1.0)
for m in MODELS:
    model = None
    try:
        model = load_model(m)
        s = MachiavelliPower(spec, str(secret)).score(model)
        res = {"model": m, "mean": round(s.mean, 4), "ci_low": round(s.ci_low, 4),
               "ci_high": round(s.ci_high, 4), "n": s.n}
    except Exception as e:  # noqa: BLE001
        res = {"model": m, "error": repr(e)}
    (out / f"res_{m.split('/')[-1]}.json").write_text(json.dumps(res, indent=2))
    print("RESULT", json.dumps(res), flush=True)
    try:
        del model
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
print("DONE", flush=True)
