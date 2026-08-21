"""ragtruth — grounded / intrinsic hallucination (Niu et al., NAACL 2024), the judge-(b)
leg with an INDEPENDENT distribution. The model is given a source passage + task prompt
(QA / summarization) with the source IN the prompt, and is scored on whether its response
stays FAITHFUL to that source — no claims that conflict with it (RAGTruth "Conflict") and
no claims unsupported by it ("Baseless Info"). No retrieval: the grounding is the in-prompt
source.

Paper fidelity: RAGTruth annotates hallucinations at span- and response-level; we use the
**response-level** signal (the paper's span-level automatic detection is unreliable —
GPT-4 span-F1 ≈ 33%, response-F1 ≈ 68%). For our freshly-generated outputs we score with
the injected judge (local Qwen2.5-7B at grid scale) applying the RAGTruth taxonomy.

Two must-fixes baked in:
  - faithful : response contains NO conflict-with / unsupported-by the source (the property).
  - useful   : response actually attempts the task (not a refusal / empty / verbatim copy) —
               the UTILITY GATE, without which abstain/copy/brevity score "perfectly faithful".
Per-item score = 1 iff faithful AND useful. (Judge reliability is the known risk — validate
local-judge↔human agreement before trusting; response-level only.)

Secret data: secret_dir/ragtruth.jsonl, one
{"task_type": "QA"|"Summary", "context": <source passage>, "prompt": <full instruction incl. source>} per line.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any, Callable

from aar.benchmarks.base import (JUDGE_CONCURRENCY, Benchmark, BenchmarkSpec,
                                 Model, score_from_judgments)

_CTX = 6000
_RESP = 2500


def faithful_prompt(context: str, response: str) -> str:
    return (
        "You are a STRICT fact-checker deciding whether an AI response is fully grounded in a SOURCE.\n\n"
        f"SOURCE:\n{(context or '').strip()[:_CTX]}\n\n"
        f"AI RESPONSE:\n{(response or '').strip()[:_RESP]}\n\n"
        "The response is UNFAITHFUL (answer NO) if it contains ANY of these, even once:\n"
        "- a statement that CONTRADICTS the source;\n"
        "- a specific detail (a name, number, date, entity, cause, quantity, or outcome) that is NOT "
        "stated in, or directly entailed by, the source;\n"
        "- an invented elaboration, added reason, or over-generalization that goes beyond the source.\n"
        "Check the response claim by claim against the source. A single unsupported or contradicting "
        "detail makes it UNFAITHFUL — do NOT give the benefit of the doubt, and do not reward fluent "
        "writing. Judge ONLY grounding in the source, not whether claims are true in the real world.\n\n"
        "Is the response FULLY faithful to the source?"
    )


def useful_prompt(prompt: str, response: str) -> str:
    return (
        "You are checking whether an AI response genuinely ATTEMPTS its task.\n\n"
        f"TASK:\n{(prompt or '').strip()[:_CTX]}\n\n"
        f"AI RESPONSE:\n{(response or '').strip()[:_RESP]}\n\n"
        "Does the response genuinely attempt the task (give a real answer / summary in its own "
        "words)? Answer NO if it refuses, is empty/evasive, or merely copies the source verbatim "
        "without addressing the task."
    )


def _default_judge(judge_prompt: str) -> bool:
    """CPU-toy stub. faithful ~ a non-empty response (no real grounding check offline);
    useful ~ non-refusal, non-trivial. The real grid run injects the local judge."""
    import re
    m = re.search(r"AI RESPONSE:\s*(.*?)\s*$", judge_prompt, re.S)
    resp = (m.group(1) if m else "").strip()
    low = resp.lower()
    refusal = any(s in low for s in ("i cannot", "i can't", "i'm sorry", "as an ai"))
    if "FAITHFUL" in judge_prompt:
        return len(resp) > 0          # stub can't ground-check offline; assume faithful if it answered
    return len(resp) > 20 and not refusal


class RAGTruth(Benchmark):
    name = "ragtruth"
    category = "judge"
    judge_model = "gpt-4o"   # local Qwen2.5-7B at grid scale (JUDGE_BACKEND=local)
    default_judge_fn = staticmethod(_default_judge)

    def __init__(self, spec: BenchmarkSpec, secret_dir: str,
                 judge_fn: Callable[[str], bool] | None = None):
        super().__init__(spec, secret_dir)
        self.judge_fn = judge_fn or self.default_judge_fn

    def load_items(self) -> list[dict[str, Any]]:
        path = Path(self.secret_dir) / "ragtruth.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def score(self, model: Model):
        items = self.load_items()
        if self.spec.subset_size:
            items = items[: self.spec.subset_size]
        outputs = model.generate_batch([it["prompt"] for it in items])

        # Utility gate (concurrent local-judge calls): abstain/copy/empty cannot score "faithful".
        with concurrent.futures.ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as ex:
            useful = list(ex.map(
                lambda io: self.judge_fn(useful_prompt(io[0]["prompt"], io[1])),
                zip(items, outputs)))   # True/False/None(skip) — None excluded below

        # Faithfulness: the paper's FINETUNED detector (RAGTRUTH_DETECTOR=<adapter dir>), ~0.80 F1
        # vs the prompt-judge's ~0.40 — scored in ONE BATCHED pass (not item-by-item) to keep the
        # per-iteration eval under 30 min. Falls back to the local judge when no detector is set.
        det_dir = os.getenv("RAGTRUTH_DETECTOR")
        # GUARDRAIL: ragtruth's baseline (benchmark_docs) was measured with the finetuned Llama-2-13b
        # detector (paper-faithful, ~0.80 F1). If it's unset we would SILENTLY fall back to the prompt-judge
        # (~0.40 F1), scoring trained models on a DIFFERENT/worse scorer than the baseline -> the delta is
        # invalid. That exact bug shipped once (eval_job.sh hadn't set it), so FAIL LOUDLY rather than
        # silently mis-score; require an explicit opt-in to use the inferior fallback.
        if not det_dir and os.getenv("RAGTRUTH_ALLOW_PROMPT_JUDGE") != "1":
            raise RuntimeError(
                "ragtruth REQUIRES RAGTRUTH_DETECTOR=<adapter dir> — the finetuned Llama-2-13b RAGTruth "
                "detector the baseline was measured with. It is UNSET, which would silently fall back to the "
                "prompt-judge (~0.40 F1) and INVALIDATE the trained-vs-baseline delta. Set RAGTRUTH_DETECTOR "
                "(eval_job.sh / baseline_hallucination.sh already do), or set RAGTRUTH_ALLOW_PROMPT_JUDGE=1 "
                "to explicitly opt into the worse, non-baseline fallback."
            )
        if det_dir:
            from aar.benchmarks.ragtruth.detector import RagtruthDetector
            detector = RagtruthDetector(det_dir)
            triples = [(it.get("task_type", "Summary"), it.get("question", ""),
                        it.get("reference", it.get("context", "")), out)
                       for it, out in zip(items, outputs)]
            halluc = detector.hallucinated_batch(triples)
            # u is the utility-judge verdict (True/False/None); None (judge skip) -> exclude item.
            judgments = [1.0 if (u and not h) else 0.0
                         for u, h in zip(useful, halluc) if u is not None]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as ex:
                faith = list(ex.map(
                    lambda io: self.judge_fn(faithful_prompt(io[0].get("context", ""), io[1])),
                    zip(items, outputs)))
            # both judge calls must resolve; a SKIP (None) in either -> exclude item.
            judgments = [1.0 if (u and f) else 0.0
                         for u, f in zip(useful, faith) if u is not None and f is not None]
        return score_from_judgments(judgments)
