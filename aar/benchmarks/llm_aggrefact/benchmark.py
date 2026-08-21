"""llm_aggrefact — LLM-AggreFact (Tang et al., "MiniCheck", EMNLP 2024; HF `lytang/LLM-AggreFact`),
a SCORED grounded-factuality leg of the hallucination axis (the 2026-06-17 factuality refocus).

Construct: grounded CLAIM VERIFICATION — given a document + a claim with the document IN the prompt,
decide whether the claim is fully SUPPORTED by the document. Aggregates 10 clean constituent
datasets (ExpertQA, TofuEval-MeetingBank/MediaSum, AggreFact-CNN/XSum, Wice, Reveal, ClaimVerify,
FactCheck-GPT, Lfqa). De-confounded from world knowledge (the evidence is provided) and from the
capability basket. Audited 2026-06-17 (grade A−/B+) on the real data + scorer.

Two REQUIRED disciplines (in the publisher, see _publish.py):
  - **EXCLUDE the RAGTruth constituent** (`dataset == "RAGTruth"`) — it is the 11th constituent and
    would overlap our existing `ragtruth` leg (item-independence).
  - **cap long documents** (some constituents have >6k-token docs) so ≤7B context isn't truncated.

Scored by single-token yes/no length-normalized logprob argmax + BALANCED ACCURACY
(constant/majority policy → 0.5) — see GroundedBinaryBenchmark. Judge-free, temperature-invariant.
"""
from __future__ import annotations

from aar.benchmarks._grounded_binary import GroundedBinaryBenchmark


class LLMAggreFact(GroundedBinaryBenchmark):
    name = "llm_aggrefact"
