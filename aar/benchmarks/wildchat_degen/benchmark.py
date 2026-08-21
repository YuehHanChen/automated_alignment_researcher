"""wildchat_degen — multi-turn FREE-FORM generation-coherence capability_filter (trajectory, RULE-graded).

Why this exists: nothing else in the loop measures **long-form, free-form, multi-turn generation coherence**.
mmlu/gsm8k are single-turn recognition; `multi_if` is multi-turn but its turns are SHORT and STRUCTURED
(IFEval instructions), so they never reach the context length where a train-short DPO fine-tune falls into
the repetition attractor. Demonstrated gap (2026-06-26): the power_seeking winner `gwhspt` PASSED `multi_if`
(0.475) yet collapsed 72-77% into token-salad on free-form multi-turn Petri. This gate closes that.

Two degeneration modes, each caught by one metric (validated on the Petri transcripts — see
`aar_overall_progress/petri_eval/`):
  - **salad / repetition collapse** — the model fills the budget with low-entropy junk (power_seeking,
    faithfulness; LONG output). Caught by normalized **token-entropy** (AUC 0.92-0.98 vs a Sonnet form-judge;
    CI-separates power_seeking + faithfulness from clean axes).
  - **terse / abstention collapse** — the model shrinks to one-line declines (concealing; winner median 8
    words, 54% < 10 words vs <=8.7% on clean axes). Caught by a **length floor**.
Per-response entropy MISSES the terse mode and a length floor MISSES the salad mode, so BOTH terms are needed
(the cross-turn-repetition term was redundant once length is included — concealing IS a length collapse).

Prompts: 250 long-form-eliciting **3-turn** threads from WildChat (`allenai/WildChat-1M`), filtered so the
reference assistant responses are >=200 words (degeneration needs ROOM to fire, and "too short" is only
meaningful when a long answer was expected) + English + non-toxic. The model is rolled through the 3 user
turns as a REAL multi-turn chat (its own responses accumulate as context), and we score **turn-3** (deepest).

Score = COHERENCE RATE = fraction of turn-3 responses that are neither terse- nor salad-collapsed, where
degenerate = (length < L_FLOOR words) OR (normalized token-entropy < H_FLOOR). Higher = better;
capability_filter floor = the base model's own coherence rate. `extra` reports a DECOMPOSITION so the AAR
sees WHICH mode binds: terse_rate, salad_rate, mean turn-3 length + entropy.

Thresholds (calibrated to the 2026-06-26 Petri output distribution): coherent long-form ~200-450 words &
entropy 0.94-0.98; salad entropy ~0.6; terse collapse < 10 words. L_FLOOR=50 / H_FLOOR=0.9 leave clear
margin and are absolute (so they hold across base models); the per-model BASELINE coherence rate is the floor.

Secret data: secret_dir/wildchat_degen.jsonl, one {"key": str, "turns": [{"prompt": str}, ...]} per line.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from aar.benchmarks.base import BenchmarkScore, Model, TrajectoryBenchmark

# Absolute degeneration thresholds (base-calibrated to the Petri distribution; see module docstring).
L_FLOOR = 50      # words: below = terse / abstention collapse. Coherent long-form is >> this (medians 200-450).
H_FLOOR = 0.9     # normalized token entropy (Pielou evenness): below = salad / repetition collapse.
                  # Sits in the empty valley between the coherent cluster (~0.94-0.98) and salad (~0.6);
                  # 0.9 hugs the coherent edge so the rate is insensitive to the exact value, and the
                  # baseline-relative gate (method CI vs base CI) makes the absolute level largely cancel.


def _norm_entropy(text: str) -> float:
    """Shannon entropy of the word distribution, normalized to [0,1] by log2(#unique words). LOW = a single
    token / short phrase dominates (repetition collapse). Length-robust via the normalization."""
    w = text.split()
    if len(w) < 2:
        return 0.0
    c = Counter(w)
    tot = len(w)
    H = -sum((v / tot) * math.log2(v / tot) for v in c.values())
    return H / math.log2(len(c)) if len(c) > 1 else 0.0


class WildchatDegen(TrajectoryBenchmark):
    name = "wildchat_degen"
    # Generation budget per turn. Long enough for a >=200-word coherent answer AND for a collapse to fill the
    # budget with salad (so the degeneration shows; EOS still bounds a healthy model's real cost).
    MAX_NEW_TOKENS = 768

    def episodes(self) -> list[dict[str, Any]]:
        path = Path(self.secret_dir) / "wildchat_degen.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def rollout(self, model: Model, episode: dict[str, Any]) -> list[dict[str, str]]:
        return self.rollout_batch(model, [episode])[0]

    def rollout_batch(self, model: Model, episodes: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
        # Real multi-turn rollout, BATCHED per turn-round across episodes (the multi_if idiom): round t feeds
        # each episode's accumulated [user/assistant ...] history + the turn-t user message as a MESSAGES
        # LIST, so HFModel renders it with the model's chat template — a genuine multi-turn chat, and the
        # model's own (possibly degenerating) outputs accumulate as context.
        histories: list[list[dict[str, str]]] = [[] for _ in episodes]
        transcripts: list[list[dict[str, str]]] = [[] for _ in episodes]
        max_turns = max((len(e["turns"]) for e in episodes), default=0)
        for t in range(max_turns):
            active = [k for k, e in enumerate(episodes) if t < len(e["turns"])]
            if not active:
                continue
            prompts = [histories[k] + [{"role": "user", "content": episodes[k]["turns"][t]["prompt"]}]
                       for k in active]
            outs = model.generate_batch(prompts, max_new_tokens=self.MAX_NEW_TOKENS)
            for k, out in zip(active, outs):
                user = {"role": "user", "content": episodes[k]["turns"][t]["prompt"]}
                asst = {"role": "assistant", "content": out}
                histories[k] += [user, asst]
                transcripts[k] += [user, asst]
        return transcripts

    @staticmethod
    def _final_response(episode: dict[str, Any], transcript: list[dict[str, str]]) -> str:
        # transcript = [user_0, asst_0, user_1, asst_1, ...]; turn t's response sits at index 2t+1.
        ri = 2 * (len(episode["turns"]) - 1) + 1
        return transcript[ri]["content"] if ri < len(transcript) else ""

    @staticmethod
    def _coherent(resp: str) -> bool:
        w = resp.split()
        if len(w) < L_FLOOR:                 # terse / abstention collapse (or no response)
            return False
        return _norm_entropy(resp) >= H_FLOOR  # else salad/repetition collapse iff entropy too low

    def grade(self, episode: dict[str, Any], transcript: list[dict[str, str]]) -> float:
        return 1.0 if self._coherent(self._final_response(episode, transcript)) else 0.0

    def score(self, model: Model) -> BenchmarkScore:
        eps = self.episodes()
        if self.spec.subset_size:
            eps = eps[: self.spec.subset_size]
        transcripts = self.rollout_batch(model, eps)
        # Per-episode turn-3 features -> headline coherence rate + a DECOMPOSITION (several metrics) so the
        # AAR sees which failure mode binds (terse vs salad), not just one scalar.
        judg: list[float] = []
        terse = salad = coherent = 0
        lens: list[int] = []
        ents: list[float] = []
        for ep, tr in zip(eps, transcripts):
            r = self._final_response(ep, tr)
            w = r.split()
            lens.append(len(w))
            if len(w) < L_FLOOR:
                terse += 1
                judg.append(0.0)
                continue
            H = _norm_entropy(r)
            ents.append(H)
            if H < H_FLOOR:
                salad += 1
                judg.append(0.0)
            else:
                coherent += 1
                judg.append(1.0)
        n = max(1, len(judg))
        # Bootstrap 95% CI on the headline coherence rate (deterministic seed for parity).
        rng = random.Random(0)
        boots = []
        for _ in range(1000):
            idx = [rng.randrange(n) for _ in range(n)]
            boots.append(sum(judg[i] for i in idx) / n)
        boots.sort()
        extra = {
            "terse_rate": round(terse / n, 4),       # fraction terse/abstention-collapsed (< L_FLOOR words)
            "salad_rate": round(salad / n, 4),       # fraction salad-collapsed (entropy < H_FLOOR)
            "mean_turn3_words": round(sum(lens) / max(1, len(lens)), 1),
            "mean_turn3_entropy": round(sum(ents) / max(1, len(ents)), 4) if ents else 0.0,
            "L_FLOOR": L_FLOOR,
            "H_FLOOR": H_FLOOR,
        }
        return BenchmarkScore(mean=coherent / n, ci_low=boots[25], ci_high=boots[974], n=n, extra=extra)


def _degenerate_self_test() -> None:
    """CI guard: coherent long text -> 1.0; terse + salad -> 0.0; reports decompose correctly."""
    coherent = " ".join(f"word{i%40}" for i in range(120))          # 120 varied words, high entropy
    terse = "I can't help with that."                                # < L_FLOOR
    salad = " ".join(["I"] * 120)                                    # long but entropy 0
    assert WildchatDegen._coherent(coherent) is True, "coherent flagged"
    assert WildchatDegen._coherent(terse) is False, "terse not caught"
    assert WildchatDegen._coherent(salad) is False, "salad not caught"
    print(f"OK  coherent_entropy={_norm_entropy(coherent):.3f}  salad_entropy={_norm_entropy(salad):.3f}")


if __name__ == "__main__":
    _degenerate_self_test()
