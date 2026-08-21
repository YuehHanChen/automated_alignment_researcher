"""bbq_intersectional — ⚠️ RETIRED as the bias held-out (2026-06-24). It was a CONCEPT shift
(intersectional/conjunctive bias ≠ the single-axis/marginal objective the methods optimize) → predicted
non-transfer, not a real method failure (root cause: aar_overall_progress/failure/bias-olmo-20260623-001635/
ROOT_CAUSE.md). The bias held-out is now `bbq_heldout` (a single-axis covariate shift). Kept registered for
reproducibility but NOT in the bias suite.

BBQ's 2 INTERSECTIONAL splits (race×gender, race×SES) of the BIAS & FAIRNESS
axis (property #6). Same construct family as `bbq` (under-specified-QA stereotype bias) but the
stereotype target is a **demographic intersection** (e.g. "the Black man" / "the Hispanic truck driver")
rather than a single axis — does the model fall back on an intersectional stereotype when the context
doesn't support it?

A one-line subclass of `BBQ`: the **identical** scorer + headline (`disambig_accuracy × (1 − |s_AMB|)`,
length-normalized answer-TEXT logprob, degenerate-proof) — only the DATA differs (the 2 intersectional
splits, item-disjoint from the 9 base categories). The intersectional stereotype-aligned answer is the
shipped `target_label` (verified: `Elfsong/BBQ` pre-computes it for the intersectional splits — e.g.
`answer_info` tags it `M-Black`), so no custom target rule is needed. Distinct on the intersection
CONSTRUCT, NOT on scorer/format (the accepted `expert_factor`/`privaci_gdpr_heldout` shared-scorer pattern)
→ a domain-generalization-style probe vs base `bbq`.

Data file: `secret_dir/bbq_intersectional.jsonl` (same schema as bbq), built by
`aar/benchmarks/bbq/_publish.py:publish_bbq_intersectional`.
"""
from __future__ import annotations

from aar.benchmarks.bbq.benchmark import BBQ


class BBQIntersectional(BBQ):
    name = "bbq_intersectional"
    # category="rule" + the disambig_acc × (1 − |s_AMB|) scorer are inherited verbatim from BBQ;
    # only `name` changes, so load_items reads bbq_intersectional.jsonl.
