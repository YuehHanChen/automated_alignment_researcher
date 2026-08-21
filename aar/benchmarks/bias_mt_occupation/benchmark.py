"""bias_mt_occupation — multi-turn, counterfactual OCCUPATION/competence bias (SCORED leg of BIAS & FAIRNESS).

WinoBias-grounded (the 40-occupation gender-stereotype vocabulary): each item puts a person with a given
occupation into a 3-turn professional-judgment scenario (initial read → leadership recommendation → final
call under a hesitant manager) and runs it as a counterfactual GENDER pair (same occupation, man vs woman).
A judge scores engagement × A-vs-B invariance; headline = engagement_rate × invariance_rate (degenerate-
proof). The second multi-turn free-form leg (distinct construct from bias_mt_decision's institutional
decisions). Scorer/rollout = aar/benchmarks/bias_mt_common.CounterfactualMTBenchmark. Data:
secret_dir/bias_mt_occupation.jsonl, built by aar/benchmarks/bias_mt_occupation/_publish.
"""
from __future__ import annotations

from aar.benchmarks.bias_mt_common import CounterfactualMTBenchmark


class BiasMTOccupation(CounterfactualMTBenchmark):
    name = "bias_mt_occupation"
