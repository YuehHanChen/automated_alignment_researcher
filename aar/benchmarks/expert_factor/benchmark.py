"""expert_factor — Expert-FACTOR (AI21Labs/factor), the HELD-OUT generalization probe.

Built from ExpertQA's specialist-domain long-form answers → contrastive factual-completion
(same clean scorer as news_factor, see _factor_common). Published with role="held_out":
scored exactly like a safety benchmark but kept OUT of the AAR-optimized headline and
stripped from every AAR-facing view (composite.strip_held_out + the publisher's held-out
tagging). It tests whether a factuality method that climbs the scored set ALSO generalizes
to an OOD knowledge *domain* (expert vs news) — same construct, different distribution.
"""
from __future__ import annotations

from aar.benchmarks._factor_common import FactorBenchmark


class ExpertFactor(FactorBenchmark):
    name = "expert_factor"
