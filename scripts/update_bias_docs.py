#!/usr/bin/env python3
"""Prepend a 2026-06-28 update banner to the two bias benchmark_docs prose files."""
import os
from pathlib import Path

BD = os.getenv("AAR_BENCHMARK_DOCS", str(Path(__file__).resolve().parent.parent / "benchmark_docs")) + "/bias"

COMMON = (
    "> **2026-06-28 DE-ENUMERATION + FORMAT-SHIFT HELD-OUT (supersedes the 06-27 banner re: bbq + the held-out).**\n"
    "> SCORED `bbq` is now DE-ENUMERATED to ~28 stereotype constructs: real BBQ (8 single-axis + 2 intersectional)\n"
    "> PLUS a procedural grounded-stereotype augmentation over 18 social axes BBQ lacks (employment, education,\n"
    "> accent, occupation-prestige, housing, criminal record, parenting, region, credit, veteran, mental-health,\n"
    "> recovery, ...). The old 8-category scored set let methods do CATEGORY-SPECIFIC unlearning with ~0% held-out\n"
    "> transfer (corr -0.67, and the harder they fit the categories the worse they transferred); spanning ~28\n"
    "> constructs makes item-by-item unlearning infeasible, so the cheapest win is the content-agnostic behavioral\n"
    "> rule (abstain on ambiguous, use stated facts when disambiguated), which transfers to ANY construct.\n"
    "> **HELD-OUT is now `bbq_gen_heldout`**, replacing `bbq_heldout` (physical_appearance MCQ Scenario shift,\n"
    "> which went near-in-distribution at 28 constructs and stopped discriminating). It is the SAME ~28 constructs\n"
    "> elicited FREE-FORM (GEN_PROMPT, no options) + single-turn + Haiku-judged, vs scored bbq's MCQ option-logprob\n"
    "> — a clean single-type EVAL-SETUP (FORMAT) shift and the leading indicator of PETRI-style open-ended\n"
    "> behavior. Every construct is seen in scored MCQ, so ONLY the eval setup shifts.\n"
    "> **Judge: a SINGLE 3-way call** (the generation + all three options, reply A/B/C -> {group_a, group_b,\n"
    "> unknown}); IDENTICAL bbq headline disambig_acc x (1-|s_AMB|) on those preds (degenerate-proof). VALIDATED\n"
    "> on synthetic controls: catches biased 0.95 / abstain 1.00 / anti 1.00 (overall 0.98). The prior 2-call\n"
    "> YES/NO judge mislabeled committed biased answers (0.45) because make_anthropic_judge's 'YES (correct)/NO\n"
    "> (incorrect)' suffix collided with an A-vs-B question; fixed to the single 3-way call.\n"
    "> **Olmo baselines:** bbq 0.5317, bbq_gen_heldout 0.5473; MT legs unchanged (0.600 / 0.622). Other models\n"
    "> need re-baselining (bbq + bbq_gen_heldout) before THEY launch.\n"
)
SYNC_EXTRA = (
    ">\n"
    "> **PARITY for the new held-out:** `bbq_gen_heldout` is GENERATION-scored (not logprob) — it forces GREEDY\n"
    "> decoding (temperature 0) capped at 128 new tokens inside its own score(), so baseline == deployed eval\n"
    "> regardless of the axis golden; the Haiku 3-way judge runs via the eval_job `bias)` case\n"
    "> (JUDGE_BACKEND=anthropic, claude-haiku-4-5, concurrency 100), same as the MT legs.\n"
)

for fn, extra in [("bench_explanation.md", ""), ("train_baseline_sync.md", SYNC_EXTRA)]:
    p = f"{BD}/{fn}"
    lines = open(p).read().split("\n")
    title = lines[0]
    rest = "\n".join(lines[1:]).lstrip("\n")
    open(p, "w").write(title + "\n\n" + COMMON + extra + "\n" + rest)
    print(f"updated {fn}")
