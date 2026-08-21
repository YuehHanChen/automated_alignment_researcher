# Prompt injection — implementation & reproduction (handoff)

Everything needed to regenerate the baselines in `baseline.json`. Written because the working
copy lives in an isolated cluster repo (the live repo's sync reverts edits to *existing* files),
so this is the durable record.

## Files

**New (additive) — committed to this repo under `aar_harness/`:**
- `aar/benchmarks/injecagent/` · `open_prompt_injection/` · `tensor_trust_hijack/` · `tensor_trust_extract/`
- `aar/benchmarks/_injecagent_src.py` — verbatim-vendored InjecAgent prompt + parser + 2-step DS.
- `scripts/run_pi_isolated.sh` (the 4-model sweep runner) and `scripts/baseline_prompt_injection.sh`.

**Edits to existing files — full validated copies in `_reference/` (these get clobbered on the live
cluster repo; land them in the canonical source):**
- `_reference/models.WITH_MESSAGES_API.py` → `aar/eval_pod/models.py`. The only change is additive:
  `generate`/`generate_batch`/`_render` (and the stub) now accept a **list of `{role,content}`
  messages** as well as a plain string (a string still = one user turn, so every other benchmark is
  unchanged). This is what lets the papers' real **system / multi-turn** roles be used.
- `_reference/publish_suite.VALIDATED.py` → `scripts/publish_suite.py`. Adds the 4 `_publish_*`
  functions + `PUBLISHERS` entries + `_SUITE_CORE["prompt_injection"]` + `_HELD_OUT["prompt_injection"]
  = "injecagent"` (2026-06-19 injecagent↔OPI swap; was `"open_prompt_injection"` pre-06-19).
  **Optimization (hill-climb) set = `{open_prompt_injection, tensor_trust_extract, tensor_trust_hijack}`.**
  injecagent = the SOLE held-out generalization probe — CROSS-SOURCE (agentic tool-call) from the
  optimized set yet genuinely predicted by it (Spearman optimization-set→injecagent **+0.42**, carried by
  the genuine tt↔injecagent ties +0.54/+0.70; the OPI leg is orthogonal to injecagent so contributes ~0),
  so it tests "does the optimized injection-resistance transfer to indirect agentic tool-injection?".
  NB the headline is a GEOMEAN, so the AAR must lift OPI too (baseline 0.313 → ample headroom).

## Reproduce the baselines

```bash
R=<repo>; PY=<venv>/bin/python; HF=<hf_cache>; HOLD=<scratch>/_pibaseline
# 1. publish the holdout (clones InjecAgent + OPI, fetches Tensor Trust, loads HF datasets):
PYTHONPATH=$R HF_HOME=$HF HF_TOKEN=$TOK $PY $R/scripts/publish_suite.py --suite prompt_injection \
  --only injecagent open_prompt_injection tensor_trust_hijack tensor_trust_extract --holdout-dir $HOLD
# 2. sweep the 4 models (Llama-3.2 + Gemma-2 dropped — see bench_explanation 'Model set'):
sbatch --array=0-3 $R/scripts/run_pi_isolated.sh     # Qwen2.5-3B, Mistral-7B-v0.3, OLMo-2-7B, Phi-3.5-mini
```
Decoding: greedy, batch 8; per-benchmark `max_new_tokens` set in each plugin (injecagent 700,
tt_extract attack 512 / DV 64, tt_hijack 64, opi 32). All rule-scored (no judge). Determinism
verified: a re-run reproduces every score byte-identically.

## Paper-fidelity (what was matched vs adapted)
Matched verbatim: OPI `system_prompts/*.txt` + `eval_<task>` (live publisher uses the **`combine`
attacker template ONLY** — strongest single attack, worst-case, 300 items; the full 5-attack set is
preserved in `_reference/publish_suite.VALIDATED.py`);
InjecAgent `SYS_PROMPT`/`USER_PROMPT` + `get_tool_dict` + `output_parser`/`evaluate_output_prompted`
+ 2-step DS (sim-response cache); Tensor Trust `medium` access-granted regex + ERR + HRR/DV.
Verified OPI's no-defense path does **no** data preprocessing (`__preprocess_data_prompt` only acts
on defense configs). Adapted (documented in bench_explanation): roles flattened where a template
rejects them (N/A for the 4 kept models); OPI = 1−ASV ungated (paper metric) with PNA-I covariate;
OPI `eval_spam` extended to accept yes/no (its `_inject` instruction asks yes/no); InjecAgent DS = the
2-step end-to-end metric.

## Validation done
4 bugs found+fixed (Mistral system-role flatten; dropped DS params/thought; OPI spam parser;
budget/truncation ruled out), CPU unit test of parse+2-step, real-model diagnostics (valid-rate,
PNA-I, per-attack ASV, raw-output eyeball), determinism re-run. See bench_explanation 'Bugs found &
fixed'. Known limitations recorded there: TT `medium` regex misses preambled grants (HRR slightly
optimistic); some TT defenses use non-"access granted" grant signals; OPI ungated metric is
competence-confounded (PNA-I covariate makes it legible).

## To finish (not done)
1. **Land the two `_reference/` edits in the canonical source** the cluster sync pulls from — else
   the eval-worker/AAR path won't have the messages API or the prompt_injection publishers.
2. Per-model baselines for `models.py`'s CI gate are not needed (these are safety baselines; the
   capability basket has its own per-model baselines).
