"""Re-baseline the 3 MACHIAVELLI power-seeking legs after the facet split (logprob, no judge → fast).
machiavelli_power=acquisition facet, machiavelli_harm=harm facet, machiavelli_heldout=all (broad, heldout
games). Auto-fallback to min_gap=1 if a facet subset yields <280 items. instrumental_eval keeps its baseline."""
import gc
import json
import sys
from pathlib import Path

from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.machiavelli_power._publish import publish_machiavelli_power
from aar.benchmarks.machiavelli_power.benchmark import MachiavelliPower
from aar.benchmarks.machiavelli_harm._publish import publish_machiavelli_harm
from aar.benchmarks.machiavelli_harm.benchmark import MachiavelliHarm
from aar.benchmarks.machiavelli_heldout._publish import publish_machiavelli_heldout
from aar.benchmarks.machiavelli_heldout.benchmark import MachiavelliHeldout
from aar.eval_pod.models import load_model

MODELS = ["Qwen/Qwen3.5-2B", "meta-llama/Llama-3.2-3B-Instruct", "allenai/Olmo-3-7B-Instruct",
          "microsoft/Phi-4-mini-instruct", "google/gemma-2-2b-it"]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mach_rebase")
OUT.mkdir(parents=True, exist_ok=True)
secret = OUT / "secret"
secret.mkdir(exist_ok=True)


def pub(fn, name, **kw):
    info = fn(str(secret / f"{name}.jsonl"), min_gap=2, **kw)
    if info["total"] < 280:
        info = fn(str(secret / f"{name}.jsonl"), min_gap=1, **kw)
        info["min_gap_used"] = 1
    else:
        info["min_gap_used"] = 2
    print(f"PUBLISH {name}:", json.dumps({k: info.get(k) for k in ("total", "facet", "candidate_items", "mean_harm_gap", "min_gap_used")}), flush=True)


pub(publish_machiavelli_power, "machiavelli_power", game_split="train", facet="acquisition")
pub(publish_machiavelli_harm, "machiavelli_harm")
pub(publish_machiavelli_heldout, "machiavelli_heldout")

LEGS = [
    ("machiavelli_power", MachiavelliPower, "safety"),
    ("machiavelli_harm", MachiavelliHarm, "safety"),
    ("machiavelli_heldout", MachiavelliHeldout, "held_out"),
]
for m in MODELS:
    model = None
    res = {"model": m}
    try:
        model = load_model(m)
        for name, cls, role in LEGS:
            spec = BenchmarkSpec(name=name, category="rule", role=role, baseline=0.0, optimum=1.0)
            s = cls(spec, str(secret)).score(model)
            res[name] = {"mean": round(s.mean, 4), "ci_low": round(s.ci_low, 4), "ci_high": round(s.ci_high, 4), "n": s.n}
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
