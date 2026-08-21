"""privaci_gdpr_heldout — the PRIVACY axis HELD-OUT generalization probe (property #7).

The GDPR regulation domain of PrivaCI-Bench (`HKUST-KnowComp/PrivaCI-Bench`), held out from the
scored `privaci_bench` set (which is the AI-Act-anchored non-GDPR remainder). Scored by the
*identical* CI-compliance scorer — verbatim `parse_decision` + logprob fallback, **guarded
prohibit-recall** (2026-06-24 redesign; inherited from `PrivaCIBench._guarded_prohibit_recall`).
The earlier 2-class macro-F1 (06-23) anti-transferred here: GDPR is prohibit-skewed (≈78:22) but the
scored set is balanced (50/50), so macro-F1's balanced optimum was a permit-ward threshold a method
could ride to win the scored set while regressing this held-out (corr −0.70). prohibit-recall is
class-prior-INVARIANT, so the scored optimum equals the held-out optimum (7-model transfer −0.39 →
+0.81; see failure/privacy-phi-20260623-193027/). The only thing that differs from the scored row is
the DATA (a different regulation) → a clean cross-regulation generalization check: does a method that
improved CI-compliance reasoning on the scored remainder ALSO transfer to GDPR?

Published with role="held_out" (publish_suite `_HELD_OUT["privacy"]`): scored exactly like a
safety benchmark but kept OUT of the AAR-optimized headline and stripped from every AAR-facing
view (composite.strip_held_out). Its FULL score is written eval-private (HELDOUT_SCORES_DIR), so
the human reads the generalization result and the AAR never sees it.

Why GDPR specifically (BENCHMARK_CATALOG §7): GDPR is the most class-imbalanced / most-memorized
domain (≈78.5% "prohibit"), so the held-out keeps GDPR's NATURAL distribution and is scored by
**prohibit-recall (= 1 − leak-rate), NEVER raw accuracy** (always-"prohibit" acc 0.785 would read as
robust; prohibit-recall is guarded by a permit-recall floor so always-"prohibit" → 0). The skewed
distribution is exactly why a prior-invariant recall metric — not macro-F1 — is the right scorer here.

Data file: `secret_dir/privaci_gdpr_heldout.jsonl` (same schema as privaci_bench), built by
`aar/benchmarks/privaci_bench/_publish.py:publish_privaci_gdpr_heldout`.
"""
from __future__ import annotations

from aar.benchmarks.privaci_bench.benchmark import PrivaCIBench


class PrivaCIGDPRHeldout(PrivaCIBench):
    name = "privaci_gdpr_heldout"
    # category="rule" + the guarded-prohibit-recall scorer + logprob fallback are inherited verbatim
    # from PrivaCIBench; only `name` changes, so load_items reads privaci_gdpr_heldout.jsonl.
