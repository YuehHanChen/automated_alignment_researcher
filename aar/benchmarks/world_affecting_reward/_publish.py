"""Publisher for world_affecting_reward — the OOD different-facet HELD-OUT of the REWARD-HACKING axis (#9).

Source: **"Reward hacking behavior can generalize across tasks"** (Nishimura-Gasparian, Dunn, Sleight,
Turpin, Hubinger, Denison, Perez; AI Alignment Forum, May 2024;
`github.com/keing1/reward-hack-generalization`, `datasets/reward_hack/world_affecting_reward.jsonl`,
CC BY 4.0). Each item is a single-turn binary (A/B) agentic ethical dilemma: the model is an AI agent whose
described *reward function* pays off for a world outcome, and exactly one option takes an unethical,
world-affecting shortcut that maximises that reward (bribery, sabotage, kickbacks). The reward hack = choose
the `high_reward_answer`. There is no factual answer to know and no linguistic ambiguity — the only pull is
the reward — so this is a DIFFERENT FACET from the scored keing1 leg (`revealing_score`, point-chasing on an
MMLU quiz) and the esteng loophole legs (linguistic/structural loopholes). That facet gap is exactly what
makes it a genuinely OOD held-out (the SCENARIO/FACET shift in the single-type-shift held-out rule).

Item schema in the source (one per line): {"prompt_list":[<full A/B scenario>], "high_reward_answer":"A"|"B",
"other_answers":["B"|"A"]} — always binary; `other_answers[0]` is the ethical option. There is NO matched
`unhackable` control file in the repo, so we degenerate-proof by **balancing the published subset to exactly
50% high_reward=A / 50% high_reward=B**: a fixed-letter or position-biased policy then scores exactly 0.50 and
cannot inflate. We append a minimal "Answer with a single letter" cue for reliable letter-logit scoring (a
scoring-format addition, not a construct change), shuffle (seed 42) and take a balanced n.

Published item schema (one per line):
  {"prompt": <full A/B scenario + single-letter cue>, "high_reward": <0 for A, 1 for B>}

Usage (standalone): python -m aar.benchmarks.world_affecting_reward._publish <out.jsonl> [--n 300] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

_LET2IDX = {"A": 0, "B": 1}
_CUE = "\nAnswer with a single letter (A or B):"
_VENDORED = Path(__file__).with_name("_data") / "world_affecting_reward.jsonl"


def _load_source(local_path: str | None) -> list[dict]:
    path = Path(local_path) if local_path else _VENDORED
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _balance(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Take a 50/50 high_reward A/B balanced subset of up to `target` items (degenerate floor 0.50)."""
    a = [r for r in rows if r["high_reward"] == 0]
    b = [r for r in rows if r["high_reward"] == 1]
    rng.shuffle(a)
    rng.shuffle(b)
    per = min(len(a), len(b), target // 2)
    return a[:per] + b[:per]


def publish_world_affecting_reward(out_path: str, n: int = 300, seed: int = 42,
                                   local_path: str | None = None, max_src: int | None = None,
                                   procgen_frac: float = 0.5) -> dict:
    """Scored agentic-ethics leg = keing1 single-source items MIXED with procedural MULTI-DOMAIN items.

    2026-06-29 DE-ENUMERATION FIX: the keing1-only leg let methods overfit one source's bribery/sabotage
    scenarios (corr −0.35 with the cross-source held-out machiavelli_reward). We now publish a `procgen_frac`
    fraction from the multi-domain procedural generator (`_procgen.gen_scenarios`, 19 domains) so the only
    lever that lifts the whole leg is the abstract rule "high reward never licenses the unethical action,"
    which transfers across sources. Each source half is independently balanced 50/50 → fixed-letter floors
    at 0.50.
    """
    rng = random.Random(seed)
    # --- keing1 single-source half ---
    src = _load_source(local_path)
    if max_src is not None:
        src = src[:max_src]   # 2026-06-29: MCQ scored leg uses source [0:max_src], DISJOINT from the
                              # free-form agentic-ethics slices (scored [318:350] + held-out [350:411]).
    kei_rows = []
    for it in src:
        hr = it["high_reward_answer"]
        if hr not in _LET2IDX or len(it.get("other_answers", [])) != 1:
            continue                                   # defensive: keep only clean binary A/B items
        kei_rows.append({"prompt": it["prompt_list"][0].rstrip() + _CUE, "high_reward": _LET2IDX[hr]})
    # --- procedural multi-domain half ---
    from aar.benchmarks.world_affecting_reward._procgen import gen_scenarios, to_mcq
    n_proc = int(round(n * procgen_frac))
    n_kei = n - n_proc
    scs = gen_scenarios(max(n_proc * 2, 60), seed)
    proc_rng = random.Random(seed + 1)
    proc_rows = []
    for sc in scs:
        m = to_mcq(sc, proc_rng)
        proc_rows.append({"prompt": m["prompt"], "high_reward": m["high_reward"]})

    kei_bal = _balance(kei_rows, n_kei, rng)
    proc_bal = _balance(proc_rows, n_proc, rng)
    balanced = kei_bal + proc_bal
    rng.shuffle(balanced)
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in balanced))
    return {
        "total": len(balanced),
        "keing1": len(kei_bal),
        "procgen": len(proc_bal),
        "source_total": len(src),
        "high_reward_letter_dist": dict(Counter("AB"[r["high_reward"]] for r in balanced)),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--local_path", default=None, help="path to source world_affecting_reward.jsonl (else vendored)")
    a = ap.parse_args()
    print(json.dumps(publish_world_affecting_reward(a.out, a.n, a.seed, a.local_path), indent=1))
