"""Publisher for wildchat_degen — sample LONG-FORM-eliciting 3-turn threads from WildChat.

Source: `allenai/WildChat-1M` (Zhao et al. 2024), real ChatGPT conversation logs. We select conversations
that are English, non-toxic, have >=3 user turns, and whose reference assistant responses are LONG-FORM
(>=`MIN_WORDS` words for >=2 of the first 3 answers). The long-form filter is the load-bearing design choice:
degeneration only manifests when the model is asked to generate at length (a short-answer prompt gives the
repetition attractor no room), and a length floor is only meaningful when a long answer was expected.

We keep the first 3 USER turns as the 3 prompts (the model generates its own responses at eval time; the
original assistant responses are used ONLY to select long-form-eliciting prompts, never shown to the model).

Output schema (one per line): {"key": conversation_hash, "turns": [{"prompt": str}, {"prompt": str}, {"prompt": str}]}
"""
from __future__ import annotations

import json
import random
from typing import Any

DATASET = "allenai/WildChat-1M"
MIN_WORDS = 200          # an assistant response is "long-form" at >= this many words
N_LONGFORM_OF_3 = 2      # require >= this many of the first 3 answers to be long-form
MAX_PROMPT_WORDS = 1200  # drop pathological mega-prompts (keeps the context budget sane)


def _wc(s: str | None) -> int:
    return len((s or "").split())


def publish_wildchat_degen(out_path: str, n: int = 250, seed: int = 42,
                           scan_limit: int = 60000) -> int:
    """Build the wildchat_degen.jsonl. Streams WildChat, applies the long-form/clean filter, takes the
    first 3 user turns, dedups by hash, and writes a seeded sample of `n`. Returns the count written."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True)
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > scan_limit:
            break
        if (row.get("language") not in (None, "English")) or row.get("toxic"):
            continue
        conv = row.get("conversation") or []
        users = [m for m in conv if m.get("role") == "user"]
        asst = [m for m in conv if m.get("role") == "assistant"]
        if len(users) < 3 or len(asst) < 3:
            continue
        # long-form: >= N_LONGFORM_OF_3 of the first 3 reference answers are >= MIN_WORDS words,
        # AND the 3rd answer (the turn we score) is itself long-form.
        first3 = asst[:3]
        if _wc(first3[2].get("content")) < MIN_WORDS:
            continue
        if sum(1 for a in first3 if _wc(a.get("content")) >= MIN_WORDS) < N_LONGFORM_OF_3:
            continue
        prompts = [u.get("content") or "" for u in users[:3]]
        if any((not p.strip()) or _wc(p) > MAX_PROMPT_WORDS for p in prompts):
            continue
        key = row.get("conversation_hash") or f"wc{scanned}"
        if key in seen:
            continue
        seen.add(key)
        picked.append({"key": key, "turns": [{"prompt": p} for p in prompts]})
        if len(picked) >= n * 3:   # over-collect, then seeded-subsample for stability
            break

    rng = random.Random(seed)
    rng.shuffle(picked)
    out = picked[:n]
    with open(out_path, "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")
    return len(out)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "wildchat_degen.jsonl"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    print("wrote", publish_wildchat_degen(out, n=n), "->", out)
