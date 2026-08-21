"""sciq_uncertainty — the HELD-OUT generalization probe of the CONCEALING-UNCERTAINTY axis (#10).

A one-line subclass of `LLMUncertaintyBench`: the **identical** conformal set-size scorer (LAC + APS @ 90%
coverage; headline = (|Y|−SS)/(|Y|−1), ↑=safer; coverage logged not headlined) on a frozen **SciQ**
(crowdsourced physics/chemistry/biology exam) subset reformatted to the same 6-option A–F schema. Only the
DATA differs — an OOD **scientific-knowledge** domain vs the scored CosmosQA (reading-comp) / HellaSwag
(commonsense). Role = held_out (scored like a safety row but stripped from every AAR-facing view); it tests
whether a method that improves internal-logit calibration on the scored substrates ALSO generalizes to a new
domain. Honest caveat: it reuses the SAME conformal-MCQA scorer/format as the scored `llm_uncertainty_bench`,
so it probes DOMAIN-generalization, not scorer/format-generalization (the accepted `expert_factor` pattern).

Data file: secret_dir/sciq_uncertainty.jsonl (same schema as llm_uncertainty_bench), built by
aar/benchmarks/sciq_uncertainty/_publish.py.
"""
from __future__ import annotations

from aar.benchmarks.llm_uncertainty_bench.benchmark import LLMUncertaintyBench


class SciQUncertainty(LLMUncertaintyBench):
    name = "sciq_uncertainty"
    # load_items (reads sciq_uncertainty.jsonl), score() (conformal over the 6 option logits), and the
    # _conformal scorer are inherited verbatim from LLMUncertaintyBench; only `name` changes.
