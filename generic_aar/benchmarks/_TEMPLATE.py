"""COPY-ME: template for a benchmark plugin on YOUR measurable task.

A benchmark is task-agnostic: it is handed a model (`generate(prompt)->str`, plus batched
variants) and returns an aggregate `BenchmarkScore`. You pick ONE of the three category base
classes and implement its one or two hooks. Set a unique `name` (class attribute) — that string
is how the suite YAML refers to it, and it auto-registers on import.

Roles (set per-entry in the suite YAML, not here):
  safety            -> a HILL-CLIMBING benchmark (the objective the AAR optimizes)
  held_out          -> the GENERALIZATION benchmark (hidden from the AAR; scored eval-private)
  capability_filter -> a gate that must not regress (needs a `floor` in the YAML)

Items may be inline (like below, zero-setup) or loaded from `self.secret_dir` (a directory of
published `<name>.jsonl` you build with a publisher — see the README for the isolated setup).
"""
from __future__ import annotations

from typing import Any

from aar.benchmarks.base import RuleBenchmark, JudgeBenchmark, TrajectoryBenchmark


# ============================================================================
# 1) RULE benchmark — deterministic match of the model's output vs a gold answer.
#    Best when correctness is a programmatic check (exact match, regex, a parser).
# ============================================================================
class MyRuleBenchmark(RuleBenchmark):
    name = "my_rule_task"          # <- unique; referenced from the suite YAML

    def load_items(self) -> list[dict[str, Any]]:
        # Each item: {"prompt": <str the model sees>, "answer": <gold used by matches()>}.
        # Return inline items, or read self.secret_dir / f"{self.name}.jsonl".
        return [
            {"prompt": "Reply with exactly the word BLUE.", "answer": "blue"},
        ]

    def matches(self, output: str, gold: Any) -> bool:
        # Deterministic correctness. Higher score = better/safer (orient it that way).
        return str(gold).lower() in output.lower()


# ============================================================================
# 2) JUDGE benchmark — an LLM judge grades free-form output vs a reference/rubric.
#    Set `judge_model` to call a real API judge (JUDGE_BACKEND=openai|anthropic),
#    and/or `default_judge_fn` for an API-free stub so the leg runs with no key.
# ============================================================================
def _stub_judge(judge_prompt: str) -> bool:
    """Deterministic, API-free judge — same control flow as a real judge, zero cost."""
    import re
    out = re.search(r"OUTPUT:\s*(.*?)\nREFERENCE:", judge_prompt, re.S)
    ref = re.search(r"REFERENCE:\s*(.*?)\s*$", judge_prompt, re.S)
    return bool(out and ref and ref.group(1).strip().lower() in out.group(1).strip().lower())


class MyJudgeBenchmark(JudgeBenchmark):
    name = "my_judge_task"
    judge_model = None                       # e.g. "gpt-4o" / "claude-haiku-4-5" for a real judge
    default_judge_fn = staticmethod(_stub_judge)   # used when no real judge is configured

    def load_items(self) -> list[dict[str, Any]]:
        return [{"prompt": "Name the capital of France. One word.", "reference": "paris"}]

    def judge_prompt(self, item: dict[str, Any], output: str) -> str:
        # Render the grading prompt. Keep an OUTPUT:/REFERENCE: block if you reuse _stub_judge.
        return f"Does the answer convey the reference?\nOUTPUT: {output}\nREFERENCE: {item['reference']}"


# ============================================================================
# 3) TRAJECTORY benchmark — multi-turn: you drive the model through an environment
#    and grade the transcript. Best for agentic / interactive tasks.
# ============================================================================
class MyTrajectoryBenchmark(TrajectoryBenchmark):
    name = "my_traj_task"

    def episodes(self) -> list[dict[str, Any]]:
        return [{"question": "What is 2+2?", "answer": "4"}]

    def rollout(self, model, episode: dict[str, Any]) -> list[dict[str, str]]:
        a = model.generate(episode["question"])
        return [{"role": "user", "content": episode["question"]},
                {"role": "assistant", "content": a}]

    def grade(self, episode: dict[str, Any], transcript: list[dict[str, str]]) -> float:
        return 1.0 if episode["answer"] in transcript[-1]["content"] else 0.0
