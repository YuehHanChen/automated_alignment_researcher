"""llm_aggrefact_B — SCORED half of the source-split LLM-AggreFact grounded claim-verification leg
(faithfulness axis). Source datasets in this half: ClaimVerify, Reveal, FactCheck-GPT, ExpertQA, Lfqa
(open-domain / long-form attributed QA + fact-checking). DISJOINT by source from `llm_aggrefact_A` (the
summarization-grounding half). Same scorer as `llm_aggrefact` (GroundedBinaryBenchmark: single-token
yes/no length-normalized logprob argmax + balanced accuracy; judge-free, temperature-invariant). Data
file is `<name>.jsonl` (llm_aggrefact_B.jsonl), published by publish_suite._publish_llm_aggrefact_B with
the B source-set filter.

See llm_aggrefact_A for why both halves are SCORED (cross-source generalization, not single-source
overfit). The held-out generalization probe for this axis is `summedits`, not this split.
"""
from __future__ import annotations

from aar.benchmarks._grounded_binary import GroundedBinaryBenchmark


class LLMAggreFactB(GroundedBinaryBenchmark):
    name = "llm_aggrefact_B"
