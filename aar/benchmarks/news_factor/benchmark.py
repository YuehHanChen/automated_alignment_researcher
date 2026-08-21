"""news_factor — News-FACTOR (AI21Labs/factor), the SCORED FACTOR leg.

Reuters/RefinedWeb news passages → contrastive factual-completion (see _factor_common).
News (not Wiki) is the scored headline: Wiki-FACTOR is built from the Pile's Wikipedia
validation split → it conflates *memorization* with factuality on models pretrained on
the Pile. Expert-FACTOR is held out (generalization probe). Family (a) logprob.
"""
from __future__ import annotations

from aar.benchmarks._factor_common import FactorBenchmark


class NewsFactor(FactorBenchmark):
    name = "news_factor"
