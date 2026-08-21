# generic_aar — the AAR harness as a template for ANY measurable task

This is the same automated-alignment-researcher harness as the parent repo, stripped to a **task-
agnostic template**. Use it to hill-climb *your own* task — anything you can measure with benchmarks.
It reuses the parent `aar/` engine unchanged (identical scoring, held-out stripping, capability gate,
submit-model loop, integrity monitor), so you only write your benchmarks + a suite file.

**The one requirement:** your task needs at least **one hill-climbing benchmark** (the objective the
AAR optimizes) and **one held-out benchmark** (a different distribution / a fresh set, to test that a
fix generalizes rather than overfits). Optional capability benchmarks act as don't-regress gates.

## Try it in 30 seconds (no GPU, no API keys)

From the repo root:

```bash
export PYTHONPATH="$PWD"
bash generic_aar/run_example.sh
# or a single run:
python generic_aar/eval.py --suite generic_aar/suite.yaml --model stub:perfect --heldout-dir generic_aar/_heldout
```

The example is an illustrative arithmetic task (so the CPU **stub** model can "solve" it): a
hill-climbing leg (`gen_task_main`, add/subtract), a second judge-scored leg (`gen_task_judge`, with an
API-free stub judge), a **held-out** leg (`gen_task_ood`, multiplication = a different distribution),
and a capability gate (`gen_capability`). `stub:perfect` scores high and generalizes to the held-out;
`stub:weak` scores low — demonstrating the whole composite end to end.

## Use it for your task — 4 steps

### 1. Write your benchmark(s)
Copy [`benchmarks/_TEMPLATE.py`](benchmarks/_TEMPLATE.py) into a new module in `benchmarks/`, pick a
category, set a unique `name`, and implement its hook(s):

| category | you implement | good for |
|---|---|---|
| **rule** | `load_items()` → `[{prompt, answer}]`, `matches(output, gold)->bool` | programmatic correctness (exact match, regex, a parser, logprob) |
| **judge** | `load_items()` → `[{prompt, reference}]`, `judge_prompt(item, output)` (+ `judge_model` and/or `default_judge_fn`) | free-form outputs graded by an LLM judge |
| **trajectory** | `episodes()`, `rollout(model, ep)`, `grade(ep, transcript)->float` | multi-turn / agentic tasks |

Then import your module in [`benchmarks/__init__.py`](benchmarks/__init__.py) so it registers. Working
references: `example_hillclimb.py` (rule), `example_judge.py` (judge, incl. an API-free stub), and the
parent `aar/benchmarks/*/benchmark.py` for real, published benchmarks. Orient every metric so
**higher = better/safer**. Items can be inline (zero-setup) or loaded from `self.secret_dir`.

### 2. Write your suite
Copy [`suite.yaml`](suite.yaml). Each entry needs `name`, `category`, and a `role`:
- `role: safety` — a **hill-climbing** benchmark (≥1 required).
- `role: held_out` — the **generalization** benchmark (≥1 required); scored eval-private and stripped
  from the AAR-facing result.
- `role: capability_filter` — a gate; add a `floor` it must not drop below.

Give each scored/held-out leg a `baseline` (your untrained model's score) and `optimum` (usually 1.0).

### 3. Measure your baselines
Run the eval with your **untrained** model and copy each benchmark's mean into `suite.yaml` as its
`baseline` (that is what `closed% = (score − baseline)/(optimum − baseline)` subtracts):

```bash
python generic_aar/eval.py --suite generic_aar/suite.yaml --model <your-untrained-hf-id> \
    --heldout-dir generic_aar/_heldout
```

### 4. Score models
Run the same command with a candidate model (a trained checkpoint dir, or the untrained id for the
baseline). Read the output:
- **`scores.json`** — `headline_pct` (geometric mean of closed-% over the `safety` legs, in percent),
  `closed_pct{<leg>: pct}`, `per_benchmark{...}`, and `passes_filter` / `filter_detail` (the capability
  gate). Held-out **excluded**.
- **`<heldout-dir>/scores.json`** — the full result: the same fields plus `held_out_pct{<leg>: pct}` and
  the held-out leg inside `per_benchmark`, kept out of the research-readable output.

Verified end-to-end (no GPU / no keys): `stub:perfect` → `headline_pct 100.0, passes_filter True`;
`stub:weak` → lower headline and `passes_filter False` (it regressed the capability gate); the held-out
leg is scored only in `<heldout-dir>/scores.json`.

### (Optional) run the full AAR loop on your task
Fill [`briefing.md`](briefing.md) for your task, then launch the loop as in the parent repo
(`python run.py server` + `python run.py agent …`; see the parent `README.md` §6 "Task B" and
`PORTABILITY.md`). The loop proposes a method → integrity monitor → train → `eval.py` scoring →
leaderboard, exactly as the 10-axis harness, but against your suite.

## Decoding / determinism / isolation — inherited from the parent
- **Decoding** is applied per benchmark from a `decoding` block (temp/top_p/seed/batch/auto_ceiling/
  no_repeat_ngram/max_new_tokens). The inline demo uses harness defaults; for a real task, ship a
  `baseline.json` with a `decoding` block and point `BENCHMARK_DOCS_DIR` at it (see the parent
  `REPRODUCE.md` §4b). This keeps baseline and trained runs identical.
- **Isolation**: keep the held-out items and `--heldout-dir` reachable only by the evaluator; the
  research side gets only `scores.json`. See the parent `ISOLATION.md` / `PORTABILITY.md`.
- **Environment**: same as the parent — the stub demo needs nothing; real models/judges need a GPU and
  the relevant judge keys (`HF_TOKEN` / `OAI_API` / `ANTHROPIC_API_KEY`).

## Layout

```
generic_aar/
  README.md                 this guide
  benchmarks/
    _TEMPLATE.py            copy-me scaffold (rule / judge / trajectory)
    example_hillclimb.py    gen_task_main  — hill-climbing (rule)
    example_judge.py        gen_task_judge — hill-climbing (judge, API-free stub)
    example_heldout.py      gen_task_ood   — held-out (rule, different distribution)
    example_capability.py   gen_capability — capability gate
    __init__.py             imports the above so they register (add yours here)
  suite.yaml                example suite (roles, baselines, optimum, floor)
  eval.py                   register benchmarks + score a model (wraps aar.eval_pod.run_eval)
  run_example.sh            no-GPU/no-key end-to-end demo
  briefing.md               AAR-loop briefing template for a custom task
```
