"""gsm8k — capability_filter rule benchmark (reasoning/math).

Generation + exact-match on the final integer answer. Part of the capability
basket (MMLU + GSM8K + IFEval) the AAR must MAINTAIN, not climb. The prompt asks
for step-by-step reasoning then a final answer (zero-shot CoT) — forcing
"final answer only" collapses GSM8K accuracy at this scale.

Secret data: secret_dir/gsm8k.jsonl, one {"prompt": str, "answer": "<int>"} per line.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from aar.benchmarks.base import RuleBenchmark

# Numbers like 1,234  -42  3.0  $5
_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str) -> str | None:
    """Pull the model's final numeric answer. Prefer the number after the last
    'answer' cue (Answer:, the answer is, ####); else the last number in the text.
    Normalizes commas/$/trailing .0 so 1,234 == 1234 and 42.0 == 42."""
    if not text:
        return None
    cue = None
    for m in re.finditer(r"(?:answer\s*(?:is|:)?|####)\s*", text, flags=re.IGNORECASE):
        cue = m.end()
    search_region = text[cue:] if cue is not None else text
    nums = _NUM.findall(search_region) or _NUM.findall(text)
    if not nums:
        return None
    return _normalize_num(nums[-1] if cue is not None else nums[-1])


def _normalize_num(tok: str) -> str:
    tok = tok.replace(",", "").replace("$", "").strip()
    try:
        f = float(tok)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return tok


class GSM8K(RuleBenchmark):
    name = "gsm8k"

    def load_items(self) -> list[dict[str, Any]]:
        path = Path(self.secret_dir) / "gsm8k.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def matches(self, output: str, gold: Any) -> bool:
        pred = extract_final_number(output)
        if pred is None:
            return False
        return pred == _normalize_num(str(gold))
