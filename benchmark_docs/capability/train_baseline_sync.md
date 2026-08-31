# capability/ — parity contract for the capability basket (mmlu, gsm8k, ifeval)

The capability basket is the **shared `capability_filter`** appended to **every** safety
axis. It is not optimized — each benchmark must clear its **per-model floor** or the
iteration is disqualified. The floor is the base model's own measured capability (CI-gated),
so it is **per-model**: a fixed floor is too lax for a stronger 7B and too strict for
gemma-2b. `baseline.json` here holds those per-model baselines (mean + 95% CI) for all 5
models; `publish_suite.py` merges them into every axis's suite, so the gate self-calibrates
to each model.

Unlike a safety axis, the capability basket has **no held-out benchmark** (nothing here is
optimized, so there is nothing to hold out as a generalization check).

## The parity contract
The capability gate compares a trained model's score to **this baseline**, so every component
that touches the score MUST be byte-identical between the baseline measurement
(`scripts/baseline_capability.sh`) and the trained-model eval (`scripts/eval_worker.sh`):

- **Decoding of the model under test:** temp-1 sampling — `strategy=sample`,
  `temperature=1.0`, `top_p=1.0`, `seed=1234` (FIXED), batch 32, AUTO token budget capped at
  `EVAL_AUTO_CEILING=4096`, no `no_repeat_ngram` guard (`EVAL_NO_REPEAT_NGRAM=0`). All three
  capability benches sit outside the free-form judge-scored gen set, so they keep
  `auto_ceiling=4096` / `no_repeat_ngram=0` (the free-form treatment of `no_repeat_ngram=4` +
  `auto_ceiling=1024` does not apply). These match the safety-suite decoding
  (`eval-decoding-parity`) — the worker scores the whole suite, capability included, with these
  settings. **Because decoding is now sampling, parity is STRICTER than under greedy: the seed
  (1234) AND batch (32) must be byte-identical between the baseline measurement and the
  trained-model eval.** Sampling is seed-dependent, and batch composition shifts the
  floating-point reduction path, so a differing seed or batch silently changes the score. (The
  old "greedy is batch-invariant, so batch is throughput-only" reasoning no longer holds.)
- **Items / subset / n:** mmlu n=300, gsm8k n=200 (CoT prompt: "Think step by step, then state
  the final answer"), ifeval n=200 (66 items needing unsupported instruction types dropped).
  Seeded subset (seed=42) via `publish_suite.py`.
- **Scorer version:** the benchmark plugin code (`aar/benchmarks/{mmlu,gsm8k,ifeval}`) — pin it.
- **No judge:** all three are rule-scored (exact-match choice / final number / instruction
  compliance) — there is no judge model to match.

If any of these changes, **re-measure** all 5 models and regenerate `baseline.json` — otherwise
the capability delta reflects the config change, not the method.

## ⚠️ Tokenizer-agnostic scoring (STANDING RULE — learned the hard way)

**The tokenizer is model-specific.** We sweep 5 models, each shipping its own tokenizer with
its own vocabulary, token IDs, and segmentation. The SAME string tokenizes differently per
model, and a token ID means different things in different models. **Any scorer that assumes a
fixed token boundary — "this candidate is one token", "compare `logits[first_token_id]`,"
hardcoded token IDs — will silently break on whichever model disagrees, and produce a
plausible-but-wrong number, not an error.**

**What bit us (2026-06-04; historical — diagnosed on the pre-2026-06-08 model set, where the
Phi slot was `Phi-3.5-mini`, since replaced by `microsoft/Phi-4-mini-instruct`):** mmlu scored
the answer by `argmax` over the **first token** of the candidates `" A"/" B"/" C"/" D"`. On most
models `" A"` is a single token, so the first token *is* the letter — fine. But the Phi model in
the set used a SentencePiece tokenizer that split `" A"` into `[▁space, "A"]` (`[29871, 319]`).
So all four candidates shared the same first token (the space, `29871`), the four "logits" were
identical, `argmax` tie-broke to index 0, and Phi **answered "A" on every question** → mmlu =
**0.203** (just the base rate of A being correct in the subset), vs its real ~0.65. It looked
like a real capability gap; it was a scorer bug. Baking that 0.203 as Phi's floor would have made
the gate meaningless for Phi. (The current `Phi-4-mini-instruct` ships its own tokenizer; don't
assume its option-letter segmentation matches any other model — verify per model, see rule 1.)

**The fix (and the rule):** score by the **full-continuation log-likelihood** — the sum of the
logprobs of *all* the tokens the candidate actually became for that model
(`HFModel.candidate_logits_batch` in `aar/eval_pod/models.py`). A shared leading-space token
contributes equally to every candidate, so it cancels in the `argmax`; the discriminating
token (the letter) decides. This is argmax-identical to first-token logits for single-token
candidates (so the single-token models are unchanged) and correct for split ones.

**Rules going forward — for ANY benchmark scored by candidate likelihood (mmlu, wei, future
multiple-choice / logprob benchmarks), across ALL models:**
1. **Never assume a candidate is a single token.** Score the whole continuation; never index
   `encode(cand)[0]` or compare a hardcoded token ID.
2. **Never compare token IDs across models** — they're not the same vocabulary.
3. **Use each model's own tokenizer / `apply_chat_template`** — never hardcode a prompt or
   answer format that happens to work for one tokenizer.
4. **Sanity-check every new model's capability scores** before trusting them: a score at the
   chance floor (≈1/n_choices for multiple-choice) is the classic signature of this bug — treat
   a near-random number as a *scorer* red flag, not a model result, and inspect the raw
   generations / candidate tokenization before recording it.

## Gate semantics
`publish_suite.py` injects each model's measured `mean` + `ci_low` into the suite YAML's
capability entries (`extra.baseline` / `extra.baseline_ci_low`). The composite gate is CI-based:
an iteration passes a capability benchmark iff its score's `ci_high >= baseline_ci_low` — i.e.
"don't measurably drop below the base model's own capability."

## Storage
`baseline.json` here is the per-model source of truth for capability floors. It carries **no
held-out information**, so unlike the per-axis `baseline.json` it is safe research-side. (The
per-axis files name the held-out and must stay eval-side mode-700 — see the top-level README.)
