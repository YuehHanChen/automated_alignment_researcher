"""bias_mt_decision — multi-turn, counterfactual DECISION bias (SCORED leg of the BIAS & FAIRNESS axis).

DiscrimEval-derived (`Anthropic/discrim-eval`, explicit): each item is one institutional decision scenario
(loan/hire/parole/medical/…) instantiated as a counterfactual demographic PAIR — the same applicant, only
race or gender swapped — and the model is rolled through 3 free-form turns (weigh factors → decide →
hold-or-cave under a colleague's pushback). A judge scores engagement × A-vs-B invariance; headline =
engagement_rate × invariance_rate (degenerate-proof). This is one of the two multi-turn free-form legs that
make the scored signal PETRI-relevant. Scorer/rollout = aar/benchmarks/bias_mt_common.CounterfactualMTBenchmark.
Data: secret_dir/bias_mt_decision.jsonl, built by aar/benchmarks/bias_mt_decision/_publish.publish_bias_mt_decision.
"""
from __future__ import annotations

from aar.benchmarks.bias_mt_common import CounterfactualMTBenchmark


class BiasMTDecision(CounterfactualMTBenchmark):
    name = "bias_mt_decision"
