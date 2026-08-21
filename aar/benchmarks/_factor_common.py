"""Shared scorer for FACTOR (Muhlgay et al., 2023; AI21Labs/factor) — contrastive
factual-completion. Each item = a prefix + 4 completions, index 0 = factual, 1-3 =
generated contradictions. The model scores every completion by its LENGTH-NORMALIZED
continuation log-prob (`model.completion_logprob_batch`, raw text / no chat template),
and the item is correct iff the factual completion (index 0) gets the HIGHEST score —
exactly AI21's `eval_factuality.py` (it sums token NLL over the completion span with the
prefix masked, divides by completion length, and takes `argmin(NLL)==0`). Accuracy over
items; random = 25%. No judge, no generation, no retrieval.

Secret data: secret_dir/<name>.jsonl, one
{"prefix": str, "completions": [factual, contradiction_0, contradiction_1, contradiction_2]}
per line (factual ALWAYS index 0).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aar.benchmarks.base import Model, RuleBenchmark, score_from_judgments


class FactorBenchmark(RuleBenchmark):
    """Base for the FACTOR domain benchmarks (news_factor / expert_factor / wiki_factor).
    A subclass just sets `name`; the data file is `<name>.jsonl` in the secret dir."""

    def load_items(self) -> list[dict[str, Any]]:
        path = Path(self.secret_dir) / f"{self.name}.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def matches(self, output: str, gold: Any) -> bool:  # unused — score() is overridden
        raise NotImplementedError

    def score(self, model: Model):
        items = self.load_items()
        if self.spec.subset_size:
            items = items[: self.spec.subset_size]
        # Flatten every (prefix, completion) pair; FACTOR scores all 4 completions per item.
        prompts: list[str] = []
        comps: list[str] = []
        ncomp: list[int] = []
        for it in items:
            cs = it["completions"]
            ncomp.append(len(cs))
            for c in cs:
                prompts.append(it["prefix"])
                comps.append(c)
        # Length-normalized mean log-prob per completion (paper-faithful; raw text).
        flat = model.completion_logprob_batch(prompts, comps, use_chat_template=False)
        judgments: list[float] = []
        k = 0
        for n in ncomp:
            sc = flat[k:k + n]
            k += n
            # correct iff the factual completion (index 0) is the argmax mean log-prob
            # (== argmin length-normalized NLL, AI21's metric).
            judgments.append(1.0 if max(range(n), key=lambda i: sc[i]) == 0 else 0.0)
        return score_from_judgments(judgments)
