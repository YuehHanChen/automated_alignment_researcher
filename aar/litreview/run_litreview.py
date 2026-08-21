"""Literature-review pre-phase.

Before a team's AARs start, a few research-librarian agents survey the literature
(via WebSearch/WebFetch) and populate the team's **shared lit forum** with >=N
structured method/paper entries — general safety-training methods AND ones
specific to the safety axis being optimized. The AARs then read this forum
(``get_literature``) to ground their designs, and may append to it.

Run (usually via scripts/litreview.sh, which sets LIT_FORUM_DIR + the API key):
    python -m aar.litreview.run_litreview --suite sycophancy --min-entries 30
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

# Sub-areas surveyed in parallel; each agent writes >= `per` entries. The {axis}
# placeholder is filled with the suite name (the safety property being optimized).
SUBAREAS = [
    ("general",
     "GENERAL safety post-training methods (model-agnostic): RLHF/RLAIF, DPO and its "
     "variants (IPO, KTO, SimPO, cDPO, ORPO), Constitutional AI / RLAIF, SFT / "
     "instruction tuning, activation steering & representation engineering (RepE, CAA, "
     "ActAdd), machine unlearning, model/representation editing, and preference-data curation."),
    ("axis-specific",
     "Methods SPECIFIC to improving {axis} in language models: find the canonical {axis} "
     "papers, datasets, and benchmarks, and the techniques proposed to mitigate {axis} "
     "failures (targeted SFT/DPO, synthetic/templated data, persona/prompting mitigations, "
     "reward-model debiasing, steering, unlearning) — cite the real papers by name."),
    ("mechanism",
     "WHY {axis} failures arise and how they are MEASURED: the training-time causes (e.g. "
     "RLHF reward-model / human-annotator biases), and the benchmarks/metrics used to "
     "evaluate {axis} (and their pitfalls)."),
    ("recent",
     "RECENT (2024-2025) and ADJACENT methods: honesty/truthfulness training, calibration, "
     "debate, self-critique / self-refine, and the instruction-following vs {axis} tradeoff."),
]

TASK_TMPL = (
    "You are a research librarian building a SHARED literature base for a team that will "
    "train small (~3B) instruct models to reduce **{axis}** while preserving capability.\n\n"
    "Survey the literature for this sub-area:\n{desc}\n\n"
    "Process:\n"
    "1. Call `get_literature` first to see what's already covered (avoid duplicates).\n"
    "2. Use **WebSearch** and **WebFetch** to find REAL papers — real titles, authors, years, "
    "arXiv/links. Read enough to summarize each correctly.\n"
    "3. For EACH distinct method or key paper (aim for **at least {per}**), read the paper "
    "properly (abstract + method + results), then call `share_literature` writing the entry as a "
    "MINI REPRODUCTION GUIDE — a researcher who never read the paper should be able to REPLICATE "
    "its main idea from your entry alone (the algorithm, the data, the objective, the key choices):\n"
    "   - method (name), category ('{cat}'), relevance (low/medium/high to {axis})\n"
    "   - summary (3-4 sentences: the problem + the idea)\n"
    "   - intuition (the KEY INSIGHT in plain terms: WHY it works, what you'd take away, and when "
    "it would or wouldn't help)\n"
    "   - core_mechanism (TECHNICAL DETAIL, 5-8 sentences: the objective/loss/algorithm, the "
    "design choices that matter, and the key idea)\n"
    "   - reproduction_recipe (an ORDERED recipe to replicate the main idea, Step 1..N: include "
    "the exact objective/loss — write the equation if there is one — and the KEY hyperparameters "
    "(lr, batch, epochs, beta/temp, optimizer, LoRA vs full). Concrete enough to actually run.)\n"
    "   - prerequisites (what you must already have: base model, any reference/reward/teacher "
    "model, preference pairs/labels, a judge, and the rough compute scale)\n"
    "   - training_data (if it TRAINS: exactly what data, how it is built/collected — human vs "
    "synthetic/templated and any generator model — size, and labels/preference pairs; if no "
    "training, say 'no training: <how it intervenes>')\n"
    "   - evaluation (the PROTOCOL to measure success so a replicator can verify: which "
    "benchmarks/datasets, the metric, any judge model, and the decoding/setup)\n"
    "   - key_results (MAIN EMPIRICAL RESULTS — concrete numbers: what improved, by how much, on "
    "which base models/benchmarks, vs which baselines, plus limitations/failure cases)\n"
    "   - applicability (how it could reduce {axis} in a ~3B instruct model, the data/compute "
    "needed, and the cost to try)\n"
    "   - source (title, authors, year, URL)\n"
    "NO REDUNDANCY: each field must add DISTINCT information — never say the same thing twice across "
    "sections. Division of labor: summary = problem+idea (high level); intuition = WHY it works (not "
    "how); core_mechanism = the conceptual how; reproduction_recipe = the concrete steps + exact "
    "loss/hyperparameters; prerequisites = inputs you need beforehand; training_data = how the data "
    "is built; evaluation = how success is measured; key_results = the numbers. Do not repeat the "
    "loss, the data, or the numbers in more than one field.\n"
    "Do NOT invent papers or fabricate numbers — only real papers you actually read, with results you "
    "actually saw. Depth over breadth: a few thoroughly-documented entries beat many shallow ones. "
    "Then end your turn."
)


async def _survey(cat: str, desc: str, axis: str, per: int, model: str, ws: Path, mcp: dict) -> None:
    from aar.research_loop.agent import BaseAgent
    from aar.research_loop.tools.lit_forum import count
    os.environ["LITREVIEW_AREA"] = cat
    allowed = ["WebSearch", "WebFetch", "Read", "Write",
               "mcp__server-api-tools__get_literature",
               "mcp__server-api-tools__share_literature"]
    agent = BaseAgent(
        name=f"lit-{cat}", allowed_tools=allowed, workspace=ws, mcp_servers=mcp, model=model,
        system_prompt="You are a careful research librarian. You cite ONLY real papers you "
                      "found via web search, with accurate titles/authors/years/links.",
    )
    task = TASK_TMPL.format(axis=axis, desc=desc.format(axis=axis), per=per, cat=cat)
    print(f"[litreview] surveying '{cat}' ...", flush=True)
    try:
        await agent.execute(task=task)
    except Exception as e:
        print(f"[litreview] '{cat}' agent error: {e}", flush=True)
    print(f"[litreview] '{cat}' done; forum now has {count()} entries", flush=True)


async def run(axis: str, min_entries: int, model: str, per: int) -> None:
    from aar.research_loop.tools.server_api_tools import create_server_api_tools_server
    from aar.research_loop.tools.lit_forum import count
    ws = Path(os.getenv("LITREVIEW_WORKSPACE", "/tmp/litreview"))
    ws.mkdir(parents=True, exist_ok=True)
    mcp = {"server-api-tools": create_server_api_tools_server()}

    # One pass over the sub-areas (sequential — keeps API/web rate sane + ordering legible).
    for cat, desc in SUBAREAS:
        await _survey(cat, desc, axis, per, model, ws, mcp)

    # Top up if we're short of the target.
    gen_desc = next(d for c, d in SUBAREAS if c == "general")
    axis_desc = next(d for c, d in SUBAREAS if c == "axis-specific")
    attempts = 0
    while count() < min_entries and attempts < 4:
        attempts += 1
        cat, desc = ("general", gen_desc) if attempts % 2 else ("axis-specific", axis_desc)
        print(f"[litreview] {count()}/{min_entries} — topping up ({cat}, attempt {attempts})", flush=True)
        await _survey(cat, desc, axis, per, model, ws, mcp)

    print(f"[litreview] DONE — {count()} entries (target {min_entries})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="sycophancy", help="safety axis / suite name")
    ap.add_argument("--min-entries", type=int, default=30)
    ap.add_argument("--per", type=int, default=9, help="target entries per sub-area agent")
    ap.add_argument("--model", default=os.getenv("LITREVIEW_MODEL", "claude-sonnet-4-6"))
    args = ap.parse_args()
    asyncio.run(run(args.suite, args.min_entries, args.model, args.per))


if __name__ == "__main__":
    main()
