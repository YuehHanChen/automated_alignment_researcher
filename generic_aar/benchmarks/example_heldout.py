"""EXAMPLE held-out benchmark (generalization test).

Same underlying skill as the hill-climbing benchmark, DIFFERENT distribution (here: multiplication
instead of add/subtract) — so a fix that truly learned the skill transfers, while one that overfit
the hill-climbing items does not. It is `role: held_out` in the suite YAML: scored eval-private and
stripped from the AAR-facing result. Illustrative arithmetic stand-in — replace with your held-out set.
"""
from __future__ import annotations

import re
from typing import Any

from aar.benchmarks.base import RuleBenchmark

_ITEMS = [
    {"prompt": "What is 6 * 8? Reply with just the number.", "answer": 48},
    {"prompt": "What is 9 * 7? Reply with just the number.", "answer": 63},
    {"prompt": "What is 12 * 5? Reply with just the number.", "answer": 60},
    {"prompt": "What is 7 * 7? Reply with just the number.", "answer": 49},
    {"prompt": "What is 11 * 6? Reply with just the number.", "answer": 66},
]


class GenTaskOOD(RuleBenchmark):
    name = "gen_task_ood"

    def load_items(self) -> list[dict[str, Any]]:
        return _ITEMS

    def matches(self, output: str, gold: Any) -> bool:
        nums = re.findall(r"-?\d+", output)
        return bool(nums) and int(nums[-1]) == int(gold)
