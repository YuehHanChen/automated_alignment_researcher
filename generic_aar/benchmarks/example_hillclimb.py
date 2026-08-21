"""EXAMPLE hill-climbing benchmark (the scored objective).

Illustrative arithmetic stand-in so the whole pipeline runs with the CPU stub model (no GPU,
no API key) — REPLACE the items + `matches` with your real task. This is the benchmark the AAR
optimizes; orient the score so higher = better.
"""
from __future__ import annotations

import re
from typing import Any

from aar.benchmarks.base import RuleBenchmark

_ITEMS = [
    {"prompt": "What is 12 + 7? Reply with just the number.", "answer": 19},
    {"prompt": "What is 34 - 9? Reply with just the number.", "answer": 25},
    {"prompt": "What is 45 + 18? Reply with just the number.", "answer": 63},
    {"prompt": "What is 81 - 27? Reply with just the number.", "answer": 54},
    {"prompt": "What is 23 + 8? Reply with just the number.", "answer": 31},
    {"prompt": "What is 60 - 14? Reply with just the number.", "answer": 46},
]


class GenTaskMain(RuleBenchmark):
    name = "gen_task_main"

    def load_items(self) -> list[dict[str, Any]]:
        return _ITEMS

    def matches(self, output: str, gold: Any) -> bool:
        nums = re.findall(r"-?\d+", output)
        return bool(nums) and int(nums[-1]) == int(gold)
