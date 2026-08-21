"""bbq_heldout — the HELD-OUT generalization probe of the BIAS & FAIRNESS axis (property #6), a
**covariate shift** that replaces the retired concept-shift `bbq_intersectional`.

It is the **single-axis BBQ category carved out of the scored set** (`physical_appearance`, never seen by the
scored `bbq`'s 8 categories), rendered with a **paraphrased template** (different preamble / field
labels / option markers — see `aar/benchmarks/bbq/_publish.py:PROMPT_HELDOUT`) and scored by the
**identical** bbq scorer (`disambig_accuracy × (1 − |s_AMB|)`, ↑=safer, degenerate-proof). So it tests
**category + template generalization** of single-axis de-biasing — the SAME mechanism the scored set
optimizes, on a category + surface it never trained on — rather than a different construct.

Honest scope (the accepted `expert_factor` / shared-scorer pattern): it reuses the bbq scorer and MCQA
answer format, so it probes category/template generalization, NOT scorer/format generalization. The
methods de-bias all axes via their training lexicon, so this is a held-out *surface* for an axis they
already de-biased; the template shift is what makes it a real generalization test rather than iid.

Role = held_out (scored like a safety row but stripped from every AAR-facing view). Data:
secret_dir/bbq_heldout.jsonl (same schema as bbq), built by
aar/benchmarks/bbq/_publish.py:publish_bbq_heldout.
"""
from __future__ import annotations

from aar.benchmarks.bbq.benchmark import BBQ


class BBQHeldout(BBQ):
    name = "bbq_heldout"
    # load_items (reads bbq_heldout.jsonl), score() and the disambig×(1−|s_AMB|) headline are inherited
    # verbatim from BBQ; only `name` changes (and the jsonl carries the paraphrased prompts + nationality
    # items).
