"""Toy RULE benchmark: 2-digit arithmetic, exact-match on the final number.

Stands in for any deterministic-scored benchmark (e.g. wei_false_math, MMLU).
Secret = the (question, answer) pairs. In production these load from
`self.secret_dir`; for the toy they're inline so it runs with zero setup.
"""
from __future__ import annotations

import re
from typing import Any

from aar.benchmarks.base import RuleBenchmark

# Held-out items (would live behind the secret S3 prefix in production).
_ITEMS = [
    {"prompt": "What is 12 + 7? Reply with just the number.", "answer": 19},
    {"prompt": "What is 34 - 9? Reply with just the number.", "answer": 25},
    {"prompt": "What is 6 * 8? Reply with just the number.", "answer": 48},
    {"prompt": "What is 45 + 18? Reply with just the number.", "answer": 63},
    {"prompt": "What is 81 - 27? Reply with just the number.", "answer": 54},
    {"prompt": "What is 9 * 9? Reply with just the number.", "answer": 81},
]


class ToyArith(RuleBenchmark):
    name = "toy_arith"

    def load_items(self) -> list[dict[str, Any]]:
        return _ITEMS

    def matches(self, output: str, gold: Any) -> bool:
        nums = re.findall(r"-?\d+", output)
        return bool(nums) and int(nums[-1]) == int(gold)
