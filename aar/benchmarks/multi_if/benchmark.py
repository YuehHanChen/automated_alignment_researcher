"""multi_if — multi-turn instruction-following capability_filter (trajectory, RULE-graded).

Multi-IF (He et al. 2024, `facebook/Multi-IF`) extends IFEval to 3 sequential turns
with CUMULATIVE verifiable instructions (turn t must satisfy the instructions of
turns 1..t). We use the ENGLISH split only. The model is rolled through the user
turns as a real multi-turn chat (its own responses accumulate as context — the exact
condition under which a multi-turn-fragile fine-tune degenerates); each turn's
response is graded DETERMINISTICALLY by the same dependency-light IFEval verifiers
the `ifeval` benchmark uses (no judge — instructions are programmatically checkable).

Score = per-turn STRICT instruction accuracy, averaged over an episode's turns
(Multi-IF's turn-level accuracy). A model that derails / repeats / collapses fails
the formatting/length/keyword checks, so the gate catches multi-turn degeneration
that single-turn ifeval/gsm8k/mmlu miss.

Secret data: secret_dir/multi_if.jsonl, one
{"key": str, "turns": [{"prompt": str, "instruction_id_list": [str], "kwargs": [dict]}, ...]} per line.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aar.benchmarks.base import BenchmarkSpec, Model, TrajectoryBenchmark, score_from_judgments
from aar.benchmarks.ifeval.benchmark import follows_all


class MultiIF(TrajectoryBenchmark):
    name = "multi_if"
    # Generation budget per turn. Multi-IF length instructions go up to ~800 words;
    # a degenerate model fills this with salad and still fails the format/length checks.
    MAX_NEW_TOKENS = 1024

    def episodes(self) -> list[dict[str, Any]]:
        path = Path(self.secret_dir) / "multi_if.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def rollout(self, model: Model, episode: dict[str, Any]) -> list[dict[str, str]]:
        return self.rollout_batch(model, [episode])[0]

    def rollout_batch(self, model: Model, episodes: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
        # Real multi-turn rollout, BATCHED per turn-round across episodes: round t feeds
        # each episode's accumulated [system?/user/assistant ...] history + the turn-t user
        # message as a MESSAGES LIST (HFModel renders it with the model's chat template, so
        # the context is a genuine multi-turn chat — not a flattened string).
        histories: list[list[dict[str, str]]] = [[] for _ in episodes]
        transcripts: list[list[dict[str, str]]] = [[] for _ in episodes]
        max_turns = max((len(e["turns"]) for e in episodes), default=0)
        for t in range(max_turns):
            active = [k for k, e in enumerate(episodes) if t < len(e["turns"])]
            if not active:
                continue
            prompts = []
            for k in active:
                user = {"role": "user", "content": episodes[k]["turns"][t]["prompt"]}
                prompts.append(histories[k] + [user])
            outs = model.generate_batch(prompts, max_new_tokens=self.MAX_NEW_TOKENS)
            for k, out in zip(active, outs):
                user = {"role": "user", "content": episodes[k]["turns"][t]["prompt"]}
                asst = {"role": "assistant", "content": out}
                histories[k] += [user, asst]
                transcripts[k] += [user, asst]
        return transcripts

    def grade(self, episode: dict[str, Any], transcript: list[dict[str, str]]) -> float | None:
        # Per-turn strict instruction accuracy, averaged over the episode's turns.
        # transcript = [user_0, asst_0, user_1, asst_1, ...]; turn t's response = index 2t+1.
        turns = episode["turns"]
        passes: list[float] = []
        for t, turn in enumerate(turns):
            ri = 2 * t + 1
            if ri >= len(transcript):
                passes.append(0.0)  # model produced no response for this turn -> fail
                continue
            resp = transcript[ri]["content"]
            kwargs = turn.get("kwargs") or [{}] * len(turn["instruction_id_list"])
            passes.append(1.0 if follows_all(resp, turn["instruction_id_list"], kwargs) else 0.0)
        if not passes:
            return None
        return sum(passes) / len(passes)
