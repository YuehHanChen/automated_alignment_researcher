"""summedits — SummEdits (Laban et al., EMNLP 2023; HF `Salesforce/summedits`), a grounded-factuality
leg of the hallucination axis (the 2026-06-17 factuality refocus).

Construct: grounded SUMMARY CONSISTENCY — given a document + a candidate summary with the document IN
the prompt, decide whether the summary is factually CONSISTENT with the document. Inconsistent items
are atomic edits (entity_modification / antonym_swap / hallucinated_fact_insertion / negation) of a
human-verified-consistent seed summary → surface-identical to consistent ones, so no length/lexical
artifact distinguishes the classes (audited 2026-06-17, grade A−/B; κ 0.72–0.90). 10 domains
(billsum/news/shakespeare/ectsum/samsum/sales_email/sales_call/podcast/scitldr/qmsumm).

Scored by single-token yes/no length-normalized logprob argmax + BALANCED ACCURACY
(constant/majority policy → 0.5) — see GroundedBinaryBenchmark. Judge-free, temperature-invariant.

NB: the audit flagged a ≤7B FLOOR RISK (the paper's ≤13B models sit near chance). Role (SCORED vs
held-out) is decided by the per-model baseline; built as a scored leg and re-classified if it floors.
"""
from __future__ import annotations

from aar.benchmarks._grounded_binary import GroundedBinaryBenchmark


class SummEdits(GroundedBinaryBenchmark):
    name = "summedits"
