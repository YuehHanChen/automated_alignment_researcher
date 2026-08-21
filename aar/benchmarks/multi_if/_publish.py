"""Publisher for multi_if — ENGLISH Multi-IF (facebook/Multi-IF), multi-turn IFEval.

Keeps only ENGLISH episodes with >=2 turns whose CUMULATIVE (last-turn) instruction
set is fully covered by our dependency-light IFEval verifiers (ifeval.SUPPORTED), so
every kept episode is graded fully-correctly. Subsamples to <=300 (capped) for a
lightweight capability gate. Each row's turn_N_prompt is a JSON {"role","content"};
turn_N_instruction_id_list / turn_N_kwargs are JSON strings (kwargs = list of JSON
dicts). Output JSONL matches multi_if/benchmark.py:episodes()."""
from __future__ import annotations

import json
import random
from pathlib import Path

N_CAP = 300  # hard cap (user constraint: n <= 300)


def _parse_turn(ex: dict, t: int) -> dict | None:
    raw_prompt = ex.get(f"turn_{t}_prompt")
    if not raw_prompt:
        return None
    content = json.loads(raw_prompt)["content"]
    iids = json.loads(ex[f"turn_{t}_instruction_id_list"])
    kw_raw = json.loads(ex[f"turn_{t}_kwargs"])
    kwargs = [json.loads(k) if isinstance(k, str) else (k or {}) for k in kw_raw]
    kwargs = [{k: v for k, v in (kw or {}).items() if v is not None} for kw in kwargs]
    return {"prompt": content, "instruction_id_list": list(iids), "kwargs": kwargs}


def publish_multi_if(out_path: str, n: int = 250, seed: int = 42) -> dict:
    from datasets import load_dataset
    from aar.benchmarks.ifeval.benchmark import item_supported

    n = min(int(n), N_CAP)
    ds = load_dataset("facebook/Multi-IF", split="train")
    idx = [i for i in range(len(ds)) if ds[i]["language"] == "English"]
    random.Random(seed).shuffle(idx)

    rows, dropped_unsupported, dropped_short = [], 0, 0
    for i in idx:
        ex = ds[i]
        turns = [tn for t in (1, 2, 3) if (tn := _parse_turn(ex, t)) is not None]
        if len(turns) < 2:
            dropped_short += 1
            continue
        # last turn's instruction list is the cumulative union; require full coverage
        if not item_supported(turns[-1]["instruction_id_list"]):
            dropped_unsupported += 1
            continue
        rows.append({"key": ex["key"], "turns": turns})
        if len(rows) >= n:
            break

    Path(out_path).write_text("\n".join(json.dumps(r) for r in rows))
    from collections import Counter
    return {
        "total": len(rows),
        "turn_dist": dict(Counter(len(r["turns"]) for r in rows)),
        "dropped_unsupported": dropped_unsupported,
        "dropped_short": dropped_short,
    }
