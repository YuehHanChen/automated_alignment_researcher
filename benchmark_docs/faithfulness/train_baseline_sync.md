> ✅ **CURRENT STATE 2026-06-27 (MT free-form redesign — supersedes the stale 2026-06-08 banner below).**
> **SCORED (5, geomean):** `ragtruth` + `llm_aggrefact_A` + `llm_aggrefact_B` (single-turn) + `faith_mt_grounded`
> + `faith_mt_claimcheck` (multi-turn free-form). **HELD-OUT:** `summedits` (unchanged). Parity for the new legs:
> - **Decoding** (both sides): sampling, `temperature=1.0`, `top_p=1.0`, `seed=1234`, batch 32 (golden
>   faithfulness config; `eval_job.sh`/`eval_worker.sh` set `suite=faithfulness`→batch 32 + the judge env below).
>   The MT engine caps generation at **`MAX_NEW_TOKENS=640`** (was 320 — truncation floored the signal; do not lower).
> - **Judge for the 2 MT legs + `ragtruth` UTILITY gate** = **`JUDGE_BACKEND=anthropic`,
>   `JUDGE_MODEL=claude-haiku-4-5`, `JUDGE_CONCURRENCY=100`** (mirrors the `bias`/`reward_hacking` case). This
>   MOVED ragtruth's utility gate local-Qwen → Haiku (benign: 0.587→0.577) → ragtruth was re-baselined on Haiku.
> - **`ragtruth` FAITHFULNESS** = the finetuned **Llama-2-13b detector** (`RAGTRUTH_DETECTOR` +
>   `RAGTRUTH_DETECTOR_BASE=meta-llama/Llama-2-13b-hf`), UNCHANGED (~0.81 F1; a general judge is only ~0.40 here).
> - **logprob legs** (`llm_aggrefact_A/B`, `summedits`) — judge-free, temperature-invariant, UNAFFECTED.
> - Per-model baselines (Llama-3.2-3B) in `baseline.json` (job 1758662); the team's `publish_holdout.sh`
>   `--target-model` bakes them into the suite YAML (verified faith_mt_claimcheck→0.10, faith_mt_grounded→0.667,
>   NOT the registry 0.40 placeholder). Re-run `baseline_faithfulness.sh <model>` if any scorer/decoding changes.
>
> ⚠️ **PARITY NOT YET LIVE (status 2026-06-08).** The baselines are now **temp-1** (refreshed from `rerun.md`),
> but the live `eval_worker.sh` still runs **greedy** — no `EVAL_TEMPERATURE`/`EVAL_TOP_P`/`EVAL_SEED`, and
> sycophancy/refusal still use batch 8 + no free-form ngram/ceiling. **A trained-vs-baseline delta computed
> right now would be INVALID.** The worker fix (set globally: `EVAL_TEMPERATURE=1.0 EVAL_TOP_P=1.0
> EVAL_SEED=1234 EVAL_RUN_BATCH=32 EVAL_RUN_NGRAM=4 EVAL_RUN_GEN_CEILING=1024`) is **PENDING a remote-code
> sync the user will trigger**. Until then, the temp-1 parity contract below is the TARGET, not live state.

# Hallucination / factuality — baseline ↔ trained-eval parity contract

> **FACTUALITY REFOCUS 2026-06-17 (see README.md).** SCORED = `ragtruth` + `llm_aggrefact` (grounded) +
> `news_factor` + `expert_factor` (FACTOR knowledge-factuality) — the full factuality picture. HELD-OUT =
> `summedits` ONLY (the generalizable grounded canary). `truthfulqa_mc2` + `truthfulqa_gen` ARCHIVED.

## `llm_aggrefact` (SCORED) + `summedits` (HELD-OUT) — shared parity surface (same scorer)
- **Judge:** none — judge-free **logprob** (length-normalized continuation log-prob argmax over the
  single-token answers "yes"/"no") with **doc-nulled contextual calibration** (`logprob(label|real) −
  logprob(label|null)`). LOGPROB is **temperature-INVARIANT** → unlike the generative legs, NO temp/seed
  pinning is needed; parity holds across decoding. The plugin caps its own batch (`LOGPROB_BATCH=8`,
  an OOM guard for large-vocab models like gemma-2-2b 256k) — batch-invariant, so baseline ↔ eval match.
- **Items / subset / n:** 300 each, frozen seed 42 (per-constituent / per-domain label-balanced); the
  `null_prompt` (document → "N/A") is frozen WITH the item. Same items + same calibration both sides.
- **Scorer version:** `aar/benchmarks/_grounded_binary.py` (shared scorer) + the two plugins + their
  `_publish.py`. Pin the commit. **RAGTruth constituent EXCLUDED** from `llm_aggrefact` (item-independence).
- **Metric: balanced accuracy** (constant policy → 0.5); optimum 1.0, per-model baseline. De-confound:
  the calibration removes the answer-prior AND the claim-plausibility (world-knowledge) confound.

## HELD-OUT benchmark: `summedits` (the GENERALIZABLE grounded canary)
- **AAR-optimized composite = `ragtruth` + `llm_aggrefact` + `news_factor` + `expert_factor`** (these four).
  FACTOR (news + expert) is now **SCORED** — knowledge-factuality is optimized directly, so a method must do
  knowledge work (e.g. knowledge-injection) to climb, not just grounding.
- **`summedits` is the sole held-out:** excluded from the optimized composite, invisible to the AAR (stripped
  from the forum / `get_leaderboard` / dashboard / `evaluate_model` return); its FULL score is written only to
  the eval-private `HELDOUT_SCORES_DIR`. Reported separately as the generalization check. (It still RUNS every
  eval — only its `role` is `held_out`.)
- **Why `summedits` is a good — and *possible* — generalization probe:** it is the **same facet** as the scored
  grounded legs (grounded-factuality recognition), so a method that improves grounded verification is *expected*
  to also move it — recognition→recognition transfer over an **OOD distribution** (summary-level, synthetic
  atomic edits, 10 domains: legal/dialogue/financial/…). That is the key contrast with the old FACTOR held-out,
  which is cross-facet (knowledge) and which grounding methods can't move (the README's −0.65) → "impossible to
  generalize." **Honest caveat:** `summedits` shares the **calibrated yes/no logprob scorer** with the scored
  `llm_aggrefact`, so it is a **domain/distribution**-generalization probe (a method that games the calibrated-
  logprob *format* could move both), NOT scorer-independent — exactly the accepted `expert_factor` /
  `privaci_gdpr_heldout` pattern (read it alongside the headline, not in isolation). A stricter, scorer-
  independent alternative would be holding out `ragtruth` (generation + detector), at the cost of a thinner
  grounded scored set — see README §"held-out choice".

## Paper-faithful setups (the source each eval matches)
- **`truthfulqa_mc2`** — Lin et al. (ACL 2022), lm-eval `process_results_mc2`: per choice = SUMMED
  continuation loglik (un-normalized) given the QA-primer + `Q:…\nA:` prompt (no chat template),
  softmax-normalize over the choice set, sum normalized mass on the `labels==1` choices. ALL 817
  validation items. Reported vs the 0.4484 uniform-spreading floor (not 0.40).
- **`truthfulqa_gen`** — Lin et al. generation task: free generation scored **truthful × informative**.
  The paper's GPT-judge/GPT-info finetunes are deprecated → LOCAL Qwen2.5-7B judge applying the SAME
  two criteria against each item's gold correct/incorrect sets. The informative gate is load-bearing
  (kills the abstain-all hack; keeps factuality≠calibration).
- **`news_factor` / `expert_factor`** — Muhlgay et al. (2023), AI21 `eval_factuality.py`: per completion,
  sum token NLL over the completion span (prefix masked) ÷ completion length, `argmin(NLL)==factual(idx0)`.
  Accuracy; random=25%. Raw text, no chat template. BOTH domains held out (News=Reuters, Expert=ExpertQA).
  (Wiki-FACTOR excluded — Pile-Wikipedia memorization confound.)
- **`ragtruth`** — Niu et al. (NAACL 2024): grounded faithfulness. Source IN the prompt; **response-level**
  (span-level auto-detection is unreliable — GPT-4 span-F1≈33% vs response-F1≈68%). FAITHFULNESS is scored
  by the paper's **FINETUNED DETECTOR** — Llama-2-13b + a LoRA we trained on RAGTruth-train (validated
  **0.808** response-level F1 vs human labels, ~0.67–0.72 on the QA+Summary tasks we use) — NOT a prompt
  judge (the local prompt-judge was only 0.40 F1; the paper explicitly shows prompt-based LLMs, even GPT-4,
  are inadequate). The detector takes the RAGTruth `[INST]`-wrapped templates, emits the
  `{"hallucination list":[...]}` JSON, batched; hallucinated = list non-empty. The **UTILITY gate** (audit
  add-on, NOT in the paper) is scored by the LOCAL Qwen2.5-7B judge: a refuse/copy/empty response cannot
  score "faithful". Per-item: faithful = (detector says no-hallucination) AND (judge says useful).

## Must-match components (baseline ↔ trained eval — byte-identical, or the delta is invalid)
| component | required value | where set |
|---|---|---|
| decoding | **sampling** (`do_sample=True`, strategy=sample) | models.py / eval env |
| temperature | **1.0** (`EVAL_TEMPERATURE=1.0`), `top_p=1.0` | both envs |
| **seed** | **`EVAL_SEED=1234`** (FIXED — byte-identical both sides; under sampling the seed IS the determinism) | both envs |
| token ceiling | logprob/detector legs (mc2/news/expert/ragtruth): `EVAL_AUTO_CEILING=4096`; free-form `truthfulqa_gen`: AUTO **1024** | both envs (`run_eval._FREEFORM_GEN`) |
| anti-repetition | OFF (`EVAL_NO_REPEAT_NGRAM=0`) everywhere EXCEPT free-form `truthfulqa_gen` → **`=4`** | both envs (`run_eval._FREEFORM_GEN`) |
| batch size | `EVAL_BATCH_SIZE=32` | both envs (baseline_hallucination.sh sets it; eval_worker.sh pins it for suite=hallucination) |
| **local judge** (truthfulqa_gen + ragtruth UTILITY gate) | `JUDGE_BACKEND=local`, `JUDGE_MODEL_LOCAL=Qwen/Qwen2.5-7B-Instruct`, temp 0 | both envs |
| **ragtruth FAITHFULNESS detector** | `RAGTRUTH_DETECTOR=<adapter dir>`, `RAGTRUTH_DETECTOR_BASE=meta-llama/Llama-2-13b-hf` (pin the LoRA adapter snapshot) | both envs |
| items / subset / n | the published holdout `.jsonl` per benchmark (no re-sampling) | holdout dir |
| scorer version | the benchmark plugin code (`aar/benchmarks/<name>/`, incl. `ragtruth/detector.py`) — pin the commit | repo |

If you change anything above for the trained eval, you must re-run `baseline.json`.

> **Why batch size is pinned (generation is not perfectly batch-invariant).** Measured on
> OLMo-2-7B *(historical; pre-2026-06-08 model refresh — that model is no longer in the set, replaced by
> `allenai/Olmo-3-7B-Instruct`)*, truthfulqa_gen under greedy decode: batch 8 → 0.6230, batch 32 → 0.6279 —
> a ~0.5 pt FP-path / batch-composition drift (the log-prob legs were identical to the digit). The finding
> still holds — and is now **stronger**: with the 2026-06-08 switch to **temp-1 sampling**, the generative
> legs (`truthfulqa_gen`) are no longer deterministic, so batch composition AND the RNG stream interact —
> which is exactly why baseline and trained eval must share BOTH `EVAL_BATCH_SIZE=32` and `EVAL_SEED=1234`
> byte-identically (the log-prob legs mc2/news/expert remain temperature- and batch-invariant). Batch 32
> was chosen for speed (+3.2×, 27→8.5 min/model, peak 36/140 GB on H200); hallucination baseline AND eval
> are pinned to **32** so the drift cancels in the trained−baseline delta. NOTE on OTHER axes: their baselines set `EVAL_BATCH_SIZE=8`
> while `eval_worker.sh` leaves it unset → trained eval runs at the models.py default **16**. This is a
> *negligible* mismatch — the real 8↔16 gap is sub-0.3 pt, well below each benchmark's ~±3 pt sampling
> CI, and only touches generation/judge legs (log-prob/rule legs are batch-invariant). Not worth a
> dedicated re-baseline; if any other axis is re-baselined for some other reason, flip its batch to 16
> then (opportunistically) to match the eval. Hallucination is the only axis where this was free to fix
> (we were re-baselining anyway for the speedup).

## ⚠️ JUDGE / SCORER PARITY (load-bearing — the composite is a delta)
The baselines were measured with the **local Qwen2.5-7B judge** (for `truthfulqa_gen` and the `ragtruth`
utility gate) **and the finetuned Llama-2-13b detector** (for `ragtruth` faithfulness). The AAR loop's
trained-model eval **MUST use the identical scorers** — i.e. `JUDGE_BACKEND=local` +
`JUDGE_MODEL_LOCAL=Qwen/Qwen2.5-7B-Instruct` + `RAGTRUTH_DETECTOR=<the same adapter>` +
`RAGTRUTH_DETECTOR_BASE=meta-llama/Llama-2-13b-hf`. If any scorer differs (e.g. swapping to gpt-4/gpt-4o
for `truthfulqa_gen`, or a different/retrained RAGTruth detector adapter), the trained−baseline **delta
reflects the scorer change, not the method, and is INVALID** — re-measure `baseline.json` under the new
scorers first. (`baseline_hallucination.sh` and `eval_worker.sh` both set these; keep them in sync.)

## Checklist before trusting a trained-vs-baseline delta
1. Same decoding env (temp-1 sampling: temperature 1.0 / top_p 1.0 / **seed 1234**; ceiling 4096 for
   logprob+detector legs, 1024+ngram4 for free-form `truthfulqa_gen`; **batch 32**) on `eval_worker.sh`
   as on the baseline (`eval_worker.sh` pins `suite=hallucination`→32 + seed 1234; `baseline_hallucination.sh`
   sets the same). Under sampling the **seed and batch must be byte-identical** both sides or the delta is invalid.
2. **Same scorers on BOTH sides** (see Parity above): `JUDGE_BACKEND=local` + `Qwen2.5-7B-Instruct`
   (truthfulqa_gen + ragtruth utility) AND the SAME `RAGTRUTH_DETECTOR` adapter + Llama-2-13b base
   (ragtruth faithfulness). Switching any scorer → re-baseline.
3. Same holdout `.jsonl` items; same plugin commit (incl. `ragtruth/detector.py`).
4. Phase-0 (already done): `truthfulqa_gen` local judge ↔ gold = 0.942; RAGTruth detector ↔ human =
   0.808 F1 (vs the prompt-judge's 0.40, which is why we use the detector). Re-validate if the adapter
   is retrained.
