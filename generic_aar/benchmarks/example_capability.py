"""EXAMPLE capability benchmark (a preserve-this gate).

`role: capability_filter` in the suite YAML with a `floor`: a method is disqualified if it regresses
this below the floor (the real harness uses a 95%-CI-below-baseline rule; a simple floor here). Use it
to hold general ability fixed while the safety legs are optimized. Illustrative arithmetic stand-in.
"""
from __future__ import annotations

import re
from typing import Any

from aar.benchmarks.base import RuleBenchmark

_ITEMS = [
    {"prompt": "What is 3 + 4? Reply with just the number.", "answer": 7},
    {"prompt": "What is 10 - 2? Reply with just the number.", "answer": 8},
    {"prompt": "What is 5 + 5? Reply with just the number.", "answer": 10},
    {"prompt": "What is 9 - 3? Reply with just the number.", "answer": 6},
]


class GenCapability(RuleBenchmark):
    name = "gen_capability"

    def load_items(self) -> list[dict[str, Any]]:
        return _ITEMS

    def matches(self, output: str, gold: Any) -> bool:
        nums = re.findall(r"-?\d+", output)
        return bool(nums) and int(nums[-1]) == int(gold)
