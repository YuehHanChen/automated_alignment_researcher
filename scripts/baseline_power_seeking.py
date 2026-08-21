"""Baseline the POWER-SEEKING (#11) axis on all 5 ≤7B models. Publishes the 3 live legs once
(machiavelli_power = train game-split; machiavelli_heldout = disjoint heldout split; instrumental_eval =
65 scenarios × pro/anti), then scores each model via the real plugins (proper BenchmarkScore w/ CIs).
Writes res_<model>.json → fold into benchmark_docs/power_seeking/baseline.json."""
import gc
import json
import sys
from pathlib import Path

from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.machiavelli_power._publish import publish_machiavelli_power
from aar.benchmarks.machiavelli_power.benchmark import MachiavelliPower
from aar.benchmarks.machiavelli_heldout._publish import publish_machiavelli_heldout
from aar.benchmarks.machiavelli_heldout.benchmark import MachiavelliHeldout
from aar.benchmarks.instrumental_eval._publish import publish_instrumental_eval
from aar.benchmarks.instrumental_eval.benchmark import InstrumentalEval
from aar.eval_pod.models import load_model

MODELS = [
    "Qwen/Qwen3.5-2B", "meta-llama/Llama-3.2-3B-Instruct", "allenai/Olmo-3-7B-Instruct",
    "microsoft/Phi-4-mini-instruct", "google/gemma-2-2b-it",
]
SCEN = sys.argv[1] if len(sys.argv) > 1 else "/opt/aar/eval-user/ie_scenarios.json"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/ps_baseline")
OUT.mkdir(parents=True, exist_ok=True)
secret = OUT / "secret"
secret.mkdir(exist_ok=True)

print("PUBLISH mp:", json.dumps(publish_machiavelli_power(str(secret / "machiavelli_power.jsonl"),
      n=300, seed=42, min_gap=2, game_split="train")), flush=True)
print("PUBLISH mh:", json.dumps(publish_machiavelli_heldout(str(secret / "machiavelli_heldout.jsonl"),
      n=300, seed=42, min_gap=2)), flush=True)
print("PUBLISH ie:", json.dumps(publish_instrumental_eval(str(secret / "instrumental_eval.jsonl"),
      scenarios_json=SCEN)), flush=True)

LEGS = [
    ("machiavelli_power", MachiavelliPower, BenchmarkSpec(name="machiavelli_power", category="rule", role="safety", baseline=0.0, optimum=1.0)),
    ("machiavelli_heldout", MachiavelliHeldout, BenchmarkSpec(name="machiavelli_heldout", category="rule", role="held_out", baseline=0.0, optimum=1.0)),
    ("instrumental_eval", InstrumentalEval, BenchmarkSpec(name="instrumental_eval", category="judge", role="safety", baseline=0.5, optimum=1.0)),
]

for m in MODELS:
    model = None
    res = {"model": m}
    try:
        model = load_model(m)
        for name, cls, spec in LEGS:
            try:
                s = cls(spec, str(secret)).score(model)
                res[name] = {"mean": round(s.mean, 4), "ci_low": round(s.ci_low, 4),
                             "ci_high": round(s.ci_high, 4), "n": s.n}
            except Exception as e:  # noqa: BLE001
                res[name] = {"error": repr(e)[:200]}
            print(f"  {m} {name}: {json.dumps(res[name])}", flush=True)
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
