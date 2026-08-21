"""Toy JUDGE benchmark: free-form answers graded by a judge.

Stands in for any judge-scored benchmark (e.g. SycEval with gpt-4o). The real
judge is an LLM over httpx; here a deterministic *stub* judge is injected so
the category runs with no API key. The stub reads the rendered judge prompt
(which embeds OUTPUT + REFERENCE) and returns whether the reference keyword
appears in the model output — same control flow as a real judge, zero cost.
"""
from __future__ import annotations

import re
from typing import Any

from aar.benchmarks.base import JudgeBenchmark

_ITEMS = [
    {"prompt": "Name the capital of France. One word.", "reference": "paris"},
    {"prompt": "What gas do plants release during photosynthesis? One word.", "reference": "oxygen"},
    {"prompt": "What is the chemical symbol for gold? One token.", "reference": "au"},
    {"prompt": "Which planet is closest to the Sun? One word.", "reference": "mercury"},
    {"prompt": "What is the largest ocean on Earth? One word.", "reference": "pacific"},
]


def stub_judge(judge_prompt: str) -> bool:
    """Deterministic, API-free judge. Parses the OUTPUT/REFERENCE block the
    benchmark rendered and returns True iff the reference keyword is present in
    the output (case-insensitive substring)."""
    out_m = re.search(r"OUTPUT:\s*(.*?)\nREFERENCE:", judge_prompt, re.S)
    ref_m = re.search(r"REFERENCE:\s*(.*?)\s*$", judge_prompt, re.S)
    if not out_m or not ref_m:
        return False
    out = out_m.group(1).strip().lower()
    ref = ref_m.group(1).strip().lower()
    return ref in out


class ToyJudge(JudgeBenchmark):
    name = "toy_judge"
    # The eval driver injects this when no real judge (gpt-4o) is configured,
    # so the judge category runs API-free in the toy.
    default_judge_fn = staticmethod(stub_judge)

    def load_items(self) -> list[dict[str, Any]]:
        return _ITEMS

    def judge_prompt(self, item: dict[str, Any], output: str) -> str:
        return (
            "Decide whether the model's answer conveys the reference answer.\n"
            f"OUTPUT: {output}\n"
            f"REFERENCE: {item['reference']}"
        )
