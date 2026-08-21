"""EXAMPLE judge-scored benchmark (a second hill-climbing leg).

Shows the JUDGE category running API-free: a deterministic stub judge is injected when no real
judge (JUDGE_BACKEND=openai|anthropic + a key) is configured. For a real task, set `judge_model`
and the harness calls that API judge; drop `default_judge_fn` if you always want a real judge.
Illustrative arithmetic stand-in — replace the items + judge_prompt with your task + rubric.
"""
from __future__ import annotations

import re
from typing import Any

from aar.benchmarks.base import JudgeBenchmark

_ITEMS = [
    {"prompt": "What is 15 + 6? Reply with just the number.", "reference": "21"},
    {"prompt": "What is 40 - 13? Reply with just the number.", "reference": "27"},
    {"prompt": "What is 19 + 24? Reply with just the number.", "reference": "43"},
    {"prompt": "What is 72 - 30? Reply with just the number.", "reference": "42"},
]


def stub_judge(judge_prompt: str) -> bool:
    """API-free judge: True iff the reference appears in the model output."""
    out = re.search(r"OUTPUT:\s*(.*?)\nREFERENCE:", judge_prompt, re.S)
    ref = re.search(r"REFERENCE:\s*(.*?)\s*$", judge_prompt, re.S)
    return bool(out and ref and ref.group(1).strip().lower() in out.group(1).strip().lower())


class GenTaskJudge(JudgeBenchmark):
    name = "gen_task_judge"
    judge_model = None                         # e.g. "gpt-4o" / "claude-haiku-4-5" for a real judge
    default_judge_fn = staticmethod(stub_judge)

    def load_items(self) -> list[dict[str, Any]]:
        return _ITEMS

    def judge_prompt(self, item: dict[str, Any], output: str) -> str:
        return (
            "Decide whether the model's answer conveys the reference answer.\n"
            f"OUTPUT: {output}\n"
            f"REFERENCE: {item['reference']}"
        )
