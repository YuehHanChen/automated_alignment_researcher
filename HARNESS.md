# AAR Harness — multi-benchmark generalization of automated-w2s-research

Forked from `automated-w2s-research` (copy-then-strip). Keeps its isolation
philosophy and agent loop; generalizes the single-dataset / PGR / predictions
contract into a **pluggable multi-benchmark suite** the AAR hillclimbs together.

## Core idea

- **Universal contract = submit-model.** A method trains a model and returns
  `{"model_path": ...}`. It never sees test inputs or answers and never emits
  predictions. A held-out **eval pod** loads the model and scores it.
- **Three benchmark categories** (`aar/benchmarks/base.py`):
  - `RuleBenchmark` — deterministic match of model outputs vs held answers.
  - `JudgeBenchmark` — an LLM judge (gpt-4o in prod; a stub in the toy) scores outputs.
  - `TrajectoryBenchmark` — server-side rollout in a held env, graded on the transcript.
- **Composite** (`benchmarks/composite.py`): headline = mean of headroom-closed
  over `safety`-role benchmarks; `capability_filter` benchmarks gate pass/fail.
- **Isolation = credential split** (Phase 2): research pods can't read the
  secret S3 prefix; only the ephemeral eval pod can.

## Layout (new in the fork)

```
aar/benchmarks/        registry + base classes + composite + toy plugins
  base.py        BenchmarkScore, BenchmarkSpec, Rule/Judge/Trajectory bases, registry
  composite.py   headroom-closed headline + capability filter
  registry.py    `python -m aar.benchmarks.registry list`
  toy_arith/     rule     (2-digit arithmetic, exact-match)
  toy_judge/     judge    (facts, stub deterministic judge — no API key)
  toy_traj/      trajectory (2-turn are-you-sure, graded by rule)
aar/eval_pod/
  models.py      load_model("stub:perfect|sycophantic|weak"); real HF = Phase 2
  run_eval.py    score a model against a suite -> scores.json
configs/toy.yaml a suite: one benchmark per category + a capability gate
ideas/TEMPLATE/run.py   the submit-model method contract + hard rules
tests/test_composite.py composite unit checks (v6 numbers as fixtures)
```

## Run the toy (CPU-only, no torch/GPU/API)

```bash
PYTHONPATH=. python3 -m aar.benchmarks.registry list
PYTHONPATH=. python3 -m aar.eval_pod.run_eval --suite configs/toy.yaml --model stub:perfect
PYTHONPATH=. python3 -m aar.eval_pod.run_eval --suite configs/toy.yaml --model stub:sycophantic
PYTHONPATH=. python3 tests/test_composite.py
```

The `sycophantic` model scores 100% on rule + judge but 0% on the trajectory
benchmark (it caves under pressure), dropping the headline +100% → +55.6% —
the demonstration that trajectory catches what answer-only scoring misses.

## Sycophancy suite — 3 safety benchmarks + the capability basket

- `benchmarks/sycophancy_eval/` — **judge** (free-response under a wrong-belief prompt; LLM judge grades factual CORRECT/INCORRECT), safety.
- `benchmarks/syc_eval/` — **trajectory** (multi-turn rebuttal pressure, judge-graded), safety.
- `benchmarks/mwe_sycophancy/` — **rule** (logprob; prefers the non-sycophantic A/B choice), safety.
- `benchmarks/{mmlu,gsm8k,ifeval}/` — **rule** capability_filter (the maintain-capability basket).

Headline = mean of headroom-closed over the 3 `safety` benchmarks; the capability basket gates pass/fail.

> Note: dataset sources, seeds, and subset sizes are deliberately NOT in this repo — the
> construction recipe (`publish_suite.py`) lives only in the eval user's mode-700 space, so the
> AAR can read the scorers (how it's graded) but cannot reconstruct the held-out items.

### Speed: where the parallelism actually lives (every iteration)
On a single GPU the GPU work serializes no matter what, so the harness
parallelizes the two things that genuinely help, and runs benchmarks
sequentially (so the cheap ones — wei/mmlu — report early instead of being
starved behind slow generation):

1. **Batched generation** — `model.generate_batch` / `candidate_logits_batch`
   process all prompts for a benchmark in chunks of `EVAL_BATCH_SIZE` (16),
   ~16× the throughput of per-item calls. syc_eval batches its two rebuttal
   turns across all episodes (2 generation rounds, not 2·N calls).
2. **Concurrent judge calls** — judge/trajectory benchmarks fan their gpt-4o
   grading out over a thread pool (`JUDGE_CONCURRENCY=12`); ~500 sequential
   calls (~8 min) collapse to ~1 min.

True *benchmark*-level parallelism (one running per GPU) is a multi-GPU story:
give the eval pod N GPUs and shard the suite across them. A shared-lock thread
pool on one GPU was tried and reverted — it starves the cheap benchmarks behind
the slow generation ones without speeding up the (serial) GPU work.

Logprob-scored rule benchmarks use the optional `Model.candidate_logits(prompt,
candidates, use_chat_template)` capability (implemented by `HFModel`; the stub
returns uniform).

Publish the secret suite, then evaluate a model against it:
```bash
# 1. publish secret holdout (run as the eval user; needs `datasets` for mmlu/syc_eval)
python scripts/publish_suite.py --suite sycophancy
#    -> $HOLDOUT_DIR/sycophancy/{wei_false_math,syc_eval,mmlu}.jsonl + sycophancy.yaml
# 2. score a trained model (real HFModel path; GPU)
python -m aar.eval_pod.run_eval \
    --suite $HOLDOUT_DIR/sycophancy/sycophancy.yaml \
    --model /path/to/trained/model --secret-dir $HOLDOUT_DIR/sycophancy
# (or, in the loop: the AAR's evaluate_model tool -> slurm_eval.sh does this)
```
Smoke-test locally with no GPU/datasets: `publish_suite.py --only wei_false_math`
then run with `--model stub:perfect` (stub gives P(No)=0.5).

## Adding a new benchmark

1. `mkdir aar/benchmarks/<name>/`, add `__init__.py` + `benchmark.py`.
2. Subclass `RuleBenchmark` / `JudgeBenchmark` / `TrajectoryBenchmark`, set
   `name`, implement the category hooks (load_items+matches / load_items+
   judge_prompt / episodes+rollout+grade). It auto-registers.
3. Add an entry to a suite YAML (`name`, `category`, `role`, `baseline`,
   `optimum` or `floor`). Done — the AAR now hillclimbs it alongside the rest.

## Transport / isolation (default = shared filesystem)

`HARNESS_TRANSPORT=fs` (default) moves the model + scores over VAST and reads
the secret suite from `HOLDOUT_DIR`; isolation is by file permissions. This
matches the Fellows cluster (no S3 is provisioned). `HARNESS_TRANSPORT=s3` is
the fallback for infra with no shared filesystem. Everything routes through
`aar/transport.py` (`put_model`/`get_model`/`resolve_suite_dir`/
`put_scores`/`read_scores`). See **ISOLATION.md** for the layout, the
separate-user hardening, and backup guidance.

## Phase 2 (wired — pending live-infra validation)

The submit-model production path is implemented end-to-end, transport-agnostic:

- `eval_pod/models.py` — `HFModel` (lazy torch/transformers) for real models; `stub:*` for the toy.
- `eval_pod/judges.py` — gpt-4o httpx judge (fail-closed); auto-used by judge benchmarks when `OAI_API` is set, else the benchmark's stub.
- `eval_pod/entrypoint.py` — runs ON the eval pod: pull model + secret suite from S3 → `run_eval` → push `scores.json` → self-terminate.
- `web_ui/backend/eval_orchestration.py` — spawns the ephemeral eval pod (`runpod.deploy_pod`) and polls S3 for `scores.json`.
- `web_ui/backend/app.py` — `/api/evaluate-model` (new) + `/api/evaluate-predictions` (legacy shim).
- `research_loop/tools/server_api_tools.py` — `evaluate_model` MCP tool (uploads model dir, calls the endpoint); added to `agent.py` allowed-tools.
- `utils/remote_evaluation.py` — `evaluate_model_remote` for non-agent callers.
- `web_ui/backend/models.py` — additive `suite` + `composite_scores` columns.
- `config.py` — `SUITE_NAME`, `SECRET_DATA_PREFIX`, `SUBMISSION_PREFIX`, `EVAL_POD_*`, `OAI_API`/`JUDGE_MODEL`.

**fs mode (default, cluster):** the `evaluate_model` tool publishes the model to
`SUBMISSIONS_DIR/<run_id>/model/`, submits `scripts/slurm_eval.sh` (which runs
`eval_pod/entrypoint.py`), and polls the shared-FS `scores.json`. No Flask
orchestrator or RunPod pod needed. **s3 mode:** the orchestrator's
`/api/evaluate-model` spawns an ephemeral RunPod eval pod instead.

**Needs validation on the cluster (can't be tested locally):**
1. Set `HARNESS_REPO`, `HOLDOUT_DIR`, `HARNESS_RUNS_DIR`, and `OAI_API` (for judge benchmarks) in the eval env / `.env`.
2. Publish a real secret suite to `HOLDOUT_DIR/<suite>/` (test inputs + answers + `<suite>.yaml`).
3. **Hardening (one admin ask):** have Eugene make `HOLDOUT_DIR` owned by a separate eval user, mode 700, and run `slurm_eval.sh` as that user — the kernel boundary then enforces both confidentiality and deletion-safety (see ISOLATION.md). Until then it's the dev honor system.
4. End-to-end: a method trains a real model → `evaluate_model` tool → `slurm_eval.sh` scores it on the secret suite → `scores.json` returns the composite.

See `~/.claude/plans/toasty-inventing-diffie.md` for the full map and `ISOLATION.md` for data-safety details.
```
