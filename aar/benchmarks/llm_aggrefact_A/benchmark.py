"""llm_aggrefact_A — SCORED half of the source-split LLM-AggreFact grounded claim-verification leg
(faithfulness axis). Source datasets in this half: AggreFact-CNN, AggreFact-XSum, TofuEval-MeetingBank,
TofuEval-MediaSum, Wice (summarization-grounding + Wikipedia claim entailment). DISJOINT by source from
`llm_aggrefact_B` (the QA/claim-verification half). Same scorer as `llm_aggrefact`
(GroundedBinaryBenchmark: single-token yes/no length-normalized logprob argmax + balanced accuracy;
judge-free, temperature-invariant). Data file is `<name>.jsonl` (llm_aggrefact_A.jsonl), published by
publish_suite._publish_llm_aggrefact_A with the A source-set filter.

WHY both halves are SCORED (not one held-out): scoring two DISJOINT source-sets separately makes the
geomean reward methods that improve grounded verification ACROSS sources, not ones that overfit a single
source distribution. The held-out generalization probe for this axis is `summedits` (a different task:
summary consistency), not this split.
"""
from __future__ import annotations

from aar.benchmarks._grounded_binary import GroundedBinaryBenchmark


class LLMAggreFactA(GroundedBinaryBenchmark):
    name = "llm_aggrefact_A"
