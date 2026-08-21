"""5-model ≤7B pilot for revealing_score (REWARD-HACKING #9). Publishes the 300-item set, scores each model
via the real plugin (candidate_logits → temperature-invariant), writes res_<model>.json. Decides scored
(all-5 headroom like the loophole legs) vs reference (ceiling/floor)."""
import gc
import json
import sys
from pathlib import Path

from aar.benchmarks.base import BenchmarkSpec
from aar.benchmarks.revealing_score._publish import publish_revealing_score
from aar.benchmarks.revealing_score.benchmark import RevealingScore
from aar.eval_pod.models import load_model

MODELS = [
    "Qwen/Qwen3.5-2B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "google/gemma-2-2b-it",
]

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rs_pilot")
out.mkdir(parents=True, exist_ok=True)
secret = out / "secret"
secret.mkdir(exist_ok=True)
local_dir = sys.argv[2] if len(sys.argv) > 2 else None   # optional local clone datasets/ dir
info = publish_revealing_score(str(secret / "revealing_score.jsonl"), n=300, seed=42, local_dir=local_dir)
print("PUBLISH", json.dumps(info), flush=True)

spec = BenchmarkSpec(name="revealing_score", category="rule", role="reference", baseline=0.5, optimum=1.0)
for m in MODELS:
    model = None
    try:
        model = load_model(m)
        s = RevealingScore(spec, str(secret)).score(model)
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
