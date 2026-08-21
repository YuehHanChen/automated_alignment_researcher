# capability/ — the capability basket (mmlu, gsm8k, ifeval)

Three general-capability benchmarks scored as **`capability_filter`** on every safety axis:
they are MAINTAINED, not climbed. A safety method that silently breaks general capability is
disqualified (gate: a benchmark's score CI must not fall clearly below the base model's). All
three are **rule-scored** (no judge). Subsets are seeded (seed=42) by `publish_suite.py`.

> **Per-axis exception (2026-06-25): `power_seeking` swaps `ifeval` → `multi_if`.** Power-seeking's
> capability basket is **`mmlu` + `gsm8k` + `multi_if`** (the MULTI-TURN instruction-following gate
> replaces the single-turn `ifeval` for this axis only; all other axes keep `ifeval`). `multi_if`
> catches multi-turn degeneration — a fine-tune that derails/repeats/collapses under accumulated
> multi-turn context — which the single-turn gates miss. It is power-seeking-specific (NOT in the
> shared basket), so its per-model baseline lives in `benchmark_docs/power_seeking/baseline.json`,
> not here. See `benchmark_docs/power_seeking/bench_explanation.md` → `multi_if`.

## mmlu — `exact_match_choice` (n=300)
- **Construct:** broad multiple-choice knowledge/reasoning (`cais/mmlu`, "all" test split).
- **Prompt:** the question + the 4 choices; the model answers with the choice.
- **Metric:** exact match of the predicted choice index vs gold. ↑ = more capable.
- **Scoring protocol:** continuation log-prob over the candidate letters `" A" " B" " C" " D"`
  after a RAW prompt (**NO chat template**) — the standard MMLU protocol; argmax of the 4
  candidate logprobs = the predicted choice (see `aar/benchmarks/mmlu/benchmark.py` +
  `candidate_logits_batch`, which sums the full continuation so multi-token letters are handled).
- **⚠️ Caveat — heavily chat-tuned models can be UNDER-scored by this protocol.** A few instruct
  models put almost no probability on the bare letter token in raw-completion format and collapse
  to a near-random, "A"-biased result, even though they answer MMLU correctly when prompted in
  CHAT format. **Observed 2026-06-07 for `google/gemma-4-E2B-it` (a model that was evaluated but
  DROPPED from the current set — historical caveat, not a current target):** standard protocol =
  **0.277** (≈random; picks "A" ~9/12 — and the candidate letters tokenize cleanly as single
  tokens, so it is NOT a tokenizer bug) vs **0.507** when scored by chat-format generation +
  letter-parse (its generations — *"The correct answer is **C**…"* — match gold on the items the
  logprob protocol missed). The current 5 models (Qwen3.5-2B 0.56 / Llama-3.2-3B 0.54 /
  Olmo-3-7B 0.57 / Phi-4-mini 0.64 / gemma-2-2b-it 0.59, temp-1) all engage with the raw protocol
  fine. **Rule of thumb:** if a model's MMLU is ~0.25–0.30 while its gsm8k/ifeval (both
  chat-format) are strong, suspect this artifact and re-measure MMLU via chat-format generation
  before trusting the cell (probe: `mmlu_chat.py` pattern — chat prompt "answer with only the
  letter" → parse A–D).

## gsm8k — `exact_match_final_number` (n=200)
- **Construct:** grade-school math word problems requiring multi-step arithmetic (`openai/gsm8k`,
  "main" test split). Stands in for the math/AMPS domain.
- **Prompt:** the problem + "Think step by step, then state the final answer." (CoT — forcing
  "final answer only" tanks gsm8k and collapses n).
- **Metric:** exact match of the extracted final integer vs gold (`#### <n>`). ↑ = more capable.

## ifeval — `instruction_compliance` (n=200)
- **Construct:** verifiable instruction-following (Google IFEval). 66 items whose instruction
  types aren't supported by the local verifier are dropped at publish time.
- **Prompt:** the instruction-bearing prompt, verbatim.
- **Metric:** fraction of the prompt's verifiable instructions the response satisfies
  (programmatic check). ↑ = better instruction-following.

## Decoding (parity with the eval worker)
Temp-1 sampling: `strategy=sample`, `temperature=1.0`, `top_p=1.0`, **`seed=1234` (FIXED)**,
**batch 32**, AUTO budget capped at 4096, no `no_repeat_ngram` guard (`no_repeat_ngram=0`) —
identical to how the eval worker scores capability during the AAR run. All three capability
benches (mmlu logprob, gsm8k math-CoT, ifeval) are OUTSIDE the free-form judge-scored gen set,
so they keep `auto_ceiling=4096` / `no_repeat_ngram=0` (the free-form treatment —
`no_repeat_ngram=4` + `auto_ceiling=1024` — does not apply here). Because decoding is now
sampling, the seed and batch are part of the parity contract, not throughput knobs. See
`train_baseline_sync.md`.
