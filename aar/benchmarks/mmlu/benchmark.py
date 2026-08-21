"""mmlu — rule benchmark (logprob-scored), used as the capability_filter.
Ported from safety_aar. Scores by argmax over the ' A'/' B'/' C'/' D'
first-token logits after the question prompt (no chat template, matching the
standard MMLU protocol). Score = accuracy.

Secret data: secret_dir/mmlu.jsonl, one
{"question": str, "choices": [4 strings], "answer": int(0-3)} per line.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aar.benchmarks.base import Model, RuleBenchmark, score_from_judgments

LETTERS = ["A", "B", "C", "D"]


def _format_prompt(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question}\n"]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


class MMLU(RuleBenchmark):
    name = "mmlu"

    def load_items(self) -> list[dict[str, Any]]:
        path = Path(self.secret_dir) / "mmlu.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def matches(self, output: str, gold: Any) -> bool:  # unused (we override score)
        raise NotImplementedError

    def score(self, model: Model):
        items = self.load_items()
        if self.spec.subset_size:
            items = items[: self.spec.subset_size]
        cand = [f" {L}" for L in LETTERS]
        prompts = [_format_prompt(it["question"], it["choices"]) for it in items]
        rows = model.candidate_logits_batch(prompts, cand, use_chat_template=False)
        judgments = [
            1.0 if max(range(len(r)), key=lambda i: r[i]) == int(it["answer"]) else 0.0
            for it, r in zip(items, rows)
        ]
        return score_from_judgments(judgments)
