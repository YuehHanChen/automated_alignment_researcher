"""Eval-pod entrypoint: score a submitted model against a secret benchmark suite.

    python -m aar.eval_pod.run_eval --suite configs/toy.yaml --model stub:perfect

Reads the suite YAML, loads the model once, runs every benchmark in the suite
(rule / judge / trajectory) via the registry, computes the composite headline +
capability filter, prints a table, and writes scores.json. Only aggregate
scores are emitted — never individual items. In production this runs on the
ephemeral eval pod; locally it runs as a plain process with a stub model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

# Parallelism. WITHIN each benchmark the expensive work already fans out (judge
# calls via JUDGE_CONCURRENCY; generation BATCHED via generate_batch). ACROSS
# benchmarks we parallelize when EVAL_GPUS>1: one model replica per GPU, the
# suite's benchmarks round-robined across GPUs and scored concurrently (the
# whole suite runs in ~its slowest benchmark instead of the sum). On a single
# GPU (EVAL_GPUS=1, default) we stay sequential — benchmark-level threads there
# just starve the cheap benchmarks behind the slow ones. Set EVAL_GPUS to the
# job's --gres gpu count (eval_worker.sh / baseline_*.sh do this).

from aar.benchmarks import registry
from aar.benchmarks.base import BenchmarkSpec, JudgeBenchmark
from aar.benchmarks.composite import compute_composite
from aar.eval_pod.models import load_model


def load_suite(path: str) -> tuple[str, list[BenchmarkSpec]]:
    data = yaml.safe_load(Path(path).read_text())
    specs = []
    for b in data["benchmarks"]:
        specs.append(BenchmarkSpec(
            name=b["name"],
            category=b["category"],
            role=b.get("role", "safety"),
            id=b.get("id", ""),
            baseline=float(b.get("baseline", 0.0)),
            optimum=float(b.get("optimum", 1.0)),
            floor=(float(b["floor"]) if "floor" in b else None),
            subset_size=b.get("subset_size"),
            extra=b.get("extra", {}) or {},
        ))
    return data.get("suite", "unnamed"), specs


def build_benchmark(spec: BenchmarkSpec, secret_dir: str, real_judge_fn=None):
    """Instantiate a benchmark plugin, wiring a judge_fn into any benchmark
    whose constructor accepts one (judge benchmarks + judge-graded trajectory
    benchmarks like syc_eval)."""
    import inspect
    cls = registry.get(spec.name)
    if "judge_fn" in inspect.signature(cls.__init__).parameters:
        # Per-paper judge (rule: use the judge model the benchmark's source paper used).
        # Each judge benchmark declares `judge_model` (e.g. sycophancy_eval/feedback ->
        # "gpt-4" per Sharma 2023; sycon_fp -> "gpt-4o" per SYCON). build_benchmark is
        # authoritative: it resolves the judge from that model, overriding any global
        # judge a caller passed, so a mixed-suite run uses the right judge per benchmark.
        jm = getattr(cls, "judge_model", None)
        judge_fn = _resolve_judge_fn(jm) or real_judge_fn or getattr(cls, "default_judge_fn", None)
        return cls(spec, secret_dir, judge_fn=judge_fn)
    return cls(spec, secret_dir)


def _resolve_judge_fn(model: str | None = None):
    """Pick the judge backend for judge-category benchmarks.

    `model` = the benchmark's declared `judge_model` (per-paper). JUDGE_BACKEND =
    "local" -> on-GPU HF judge (grid default; bounds cost — but NOT paper-faithful,
    so not used for the published baselines). "openai"/unset -> the per-paper model
    (falls back to JUDGE_MODEL env then gpt-4o) if a key is present, else None.
    """
    import os
    backend = os.getenv("JUDGE_BACKEND", "openai").lower()
    if backend == "local":
        from aar.eval_pod.judges import make_local_judge
        return make_local_judge(os.getenv("JUDGE_MODEL_LOCAL") or None)
    if backend == "anthropic":
        from aar.eval_pod.judges import make_anthropic_judge
        from aar.benchmarks._judge_http import _anthropic_key
        if _anthropic_key():
            return make_anthropic_judge(model=(os.getenv("JUDGE_MODEL") or "claude-haiku-4-5"))
        return None
    if os.getenv("OAI_API") or os.getenv("OPENAI_API_KEY"):
        from aar.eval_pod.judges import make_openai_judge
        return make_openai_judge(model=(model or os.getenv("JUDGE_MODEL") or "gpt-4o"))
    return None


# Benchmarks whose baseline is the SHARED capability basket (benchmark_docs/capability/),
# not the safety axis. They must be scored at the capability config on EVERY axis.
_CAP_BENCHES = {"mmlu", "gsm8k", "ifeval"}
_DECODING_CACHE: dict[str, dict] = {}

# FREE-FORM GENERATION benchmarks: long, open-ended completions where a degenerate model can
# fall into a no-EOS repetition loop. These (and only these) get the optional run-time
# no_repeat_ngram guard + tighter ceiling (EVAL_RUN_NGRAM / EVAL_RUN_GEN_CEILING below).
# EXCLUDED on purpose:
#   - ifeval: instructions REQUIRE legit repetition / verbatim echoing / keyword-frequency — an
#     ngram guard makes compliance impossible and tanks the score.
#   - gsm8k: math CoT with an EXACT final-answer format. A tight ceiling truncates verbose chains
#     before the "#### answer" (Qwen 0.76 -> 0.055) and no_repeat_ngram disrupts the arithmetic
#     (math naturally repeats number/operator n-grams; Llama 0.665 -> 0.40). It is a reasoning
#     task, NOT free-form text — keep it at the doc config (ceiling 4096, ngram 0).
#   - logprob/MCQ benches (mmlu, truthfulqa_mc2, news_factor, expert_factor): no generation -> no-op.
#   - structured prompt-injection extraction benches: short exact-string outputs ngram could corrupt.
# CAUTION for later: ragtruth measures source-FAITHFULNESS; an ngram guard could force the model to
# diverge from the source it should quote -> revisit/validate before trusting task-3 ragtruth.
_FREEFORM_GEN = {
    # JUDGE-scored, short-response free-form benches only (the judge is robust to the ngram
    # guard). EXCLUDED after the hard way: rule-scored pattern-keyed benches (gsm8k arithmetic,
    # ifeval echoing, elephant_aita affirmation-phrases — ngram inflates them), trajectory/long
    # benches (sycon_fp — ceiling-1024 truncates the multi-turn rollout), and detector-scored
    # faithfulness (ragtruth — ngram could force divergence from the quoted source).
    "sycophancy_eval", "sycophancy_feedback",
    "truthfulqa_gen",
    "mask_generative", "mask_factual", "deceptionbench_pressure", "deceptionbench_reward",
    "harmbench", "jbb", "jbb_artifacts", "strongreject",
    # NB: sycon_fp (trajectory held-out) was TEMPORARILY added here for its dedicated temp-1 re-run
    # (array 1669675, 2026-06-08) so EVAL_RUN_NGRAM/CEILING could bound its degeneration runaways;
    # that run is DONE and it is now removed — it must NOT carry the ngram/ceiling treatment normally.
}


def _run_override(spec: BenchmarkSpec, d: dict) -> dict:
    """Report-only re-measurement overrides (see rerun config), env-gated so the live AAR's
    golden configs are untouched when these envs are unset:
      EVAL_RUN_BATCH       -> force batch_size for ALL benches (uniform throughput)
      EVAL_RUN_NGRAM       -> no_repeat_ngram for FREE-FORM GEN benches only (kills runaways)
      EVAL_RUN_GEN_CEILING -> auto_ceiling for FREE-FORM GEN benches only (caps the tail)
    Temperature is carried by the model (EVAL_TEMPERATURE) and is not touched here."""
    bs = os.getenv("EVAL_RUN_BATCH")
    if bs:
        d["batch_size"] = int(bs)
    if spec.name in _FREEFORM_GEN:
        ng = os.getenv("EVAL_RUN_NGRAM")
        if ng:
            d["no_repeat_ngram"] = int(ng)
        cg = os.getenv("EVAL_RUN_GEN_CEILING")
        if cg:
            d["auto_ceiling"] = int(cg)
    return d


def _golden_decoding(spec: BenchmarkSpec, suite_name: str) -> dict:
    """The benchmark's GOLDEN decoding, read live from benchmark_docs/<group>/baseline.json
    — the SAME config its baseline was measured at. The shared capability basket maps to
    'capability' (fixed across all axes); every other benchmark maps to the suite's axis.
    Returns {} when unavailable, so we keep the current/global decoding as a safe fallback."""
    import json as _json
    from aar import config as _cfg
    group = "capability" if spec.name in _CAP_BENCHES else suite_name
    if group not in _DECODING_CACHE:
        d: dict = {}
        try:
            p = Path(_cfg.BENCHMARK_DOCS_DIR) / group / "baseline.json"
            dec = (_json.loads(p.read_text()).get("decoding") or {})
            d = {"batch_size": dec.get("batch", dec.get("batch_size")),
                 "auto_ceiling": dec.get("auto_ceiling"),
                 "no_repeat_ngram": dec.get("no_repeat_ngram", 0),
                 "max_new_tokens": dec.get("max_new_tokens"),
                 # decoding STRATEGY from the docs (the config the baseline was measured at):
                 # temp-1 sampling (temperature/top_p) + a fixed seed for reproducibility.
                 "temperature": dec.get("temperature"),
                 "top_p": dec.get("top_p"),
                 "seed": dec.get("seed"),
                 # free-form judge-scored gen benches get the doc's freeform_treatment override
                 # (tighter ngram/ceiling) — applied per-benchmark below, not baked into the cache.
                 "_freeform": dec.get("freeform_treatment") or {}}
        except Exception as e:
            print(f"  [decoding] WARN: no benchmark_docs decoding for group '{group}' "
                  f"({e}); keeping global decoding", flush=True)
        _DECODING_CACHE[group] = d
    d = dict(_DECODING_CACHE[group])
    ff = d.pop("_freeform", {}) or {}
    if spec.name in _FREEFORM_GEN:
        if ff.get("no_repeat_ngram") is not None:
            d["no_repeat_ngram"] = ff["no_repeat_ngram"]
        if ff.get("auto_ceiling") is not None:
            d["auto_ceiling"] = ff["auto_ceiling"]
    return d


def _apply_golden(model, spec: BenchmarkSpec, suite_name: str) -> None:
    """Pin the model's decoding to THIS benchmark's golden config before scoring it, so
    per-iteration eval == baseline measurement for every benchmark (the capability basket
    at batch8/ceiling4096/ngram0 on every axis; safety/held-out at the axis's config)."""
    if not hasattr(model, "apply_decoding"):
        return  # toy/stub model: no real decoding to set
    d = _golden_decoding(spec, suite_name)
    if not d:
        return
    d = _run_override(spec, d)   # report-only run overrides (env-gated; no-op for the live AAR)
    kw: dict = {}
    if d.get("batch_size") is not None:
        kw["batch_size"] = d["batch_size"]
    if d.get("no_repeat_ngram") is not None:
        kw["no_repeat_ngram"] = d["no_repeat_ngram"]
    if d.get("auto_ceiling") is not None:
        kw["auto_ceiling"] = d["auto_ceiling"]
    if d.get("temperature") is not None:
        kw["temperature"] = d["temperature"]
    if d.get("top_p") is not None:
        kw["top_p"] = d["top_p"]
    if d.get("seed") is not None:
        kw["seed"] = d["seed"]
    # Explicit max_new_tokens (scalar, or a per-benchmark map e.g. prompt_injection) overrides
    # AUTO; otherwise use AUTO bounded by the group ceiling (resetting any prior explicit value).
    mnt = d.get("max_new_tokens")
    if isinstance(mnt, dict):
        mnt = mnt.get(spec.id) or mnt.get(spec.name)
    if mnt is not None:
        kw["max_new_tokens"] = mnt
    elif d.get("auto_ceiling") is not None:
        kw["max_new_tokens"] = None
    if kw:
        model.apply_decoding(**kw)
        grp = "capability" if spec.name in _CAP_BENCHES else suite_name
        print(f"  [decoding] {spec.id} ({grp}): batch={model.batch_size} "
              f"ceiling={model._auto_ceiling} ngram={model._no_repeat_ngram} "
              f"max_new={model.max_new_tokens} temp={model._temperature} "
              f"top_p={model._top_p} seed={model._seed}", flush=True)


def _score_sequential(specs, model_ref, secret_dir, suite_name=""):
    model = load_model(model_ref)
    judge_fn = _resolve_judge_fn()
    scores = {}
    for spec in specs:
        _apply_golden(model, spec, suite_name)   # golden per-benchmark decoding (== baseline)
        sc = build_benchmark(spec, secret_dir, real_judge_fn=judge_fn).score(model)
        scores[spec.id] = sc
        if spec.role == "held_out":
            # Never print the held-out score to stdout — eval logs can land in
            # research-readable paths; the only place its score is allowed is the
            # eval-private HELDOUT_SCORES_DIR.
            print("  [a held_out benchmark — name, score, n all hidden from logs]", flush=True)
        else:
            print(f"  {spec.id:16s} [{spec.category:10s} {spec.role:18s}] "
                  f"mean={sc.mean:.4f} ci=[{sc.ci_low:.4f},{sc.ci_high:.4f}] n={sc.n}", flush=True)
    return scores


def _gpu_worker(gpu: str, specs, model_ref: str, secret_dir: str, q, suite_name="") -> None:
    """One GPU's process: pin the GPU, load a model replica, score its assigned
    benchmarks, push each result to the queue. Errors are pushed (not raised) so
    the parent never hangs."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        from aar.benchmarks import registry as _reg
        _reg.discover()
        judge_fn = _resolve_judge_fn()
        model = load_model(model_ref)
        for spec in specs:
            try:
                _apply_golden(model, spec, suite_name)   # golden per-benchmark decoding (== baseline)
                sc = build_benchmark(spec, secret_dir, real_judge_fn=judge_fn).score(model)
                q.put((spec.id, sc, None))
            except Exception as e:  # one benchmark failing shouldn't sink the rest
                q.put((spec.id, None, repr(e)))
    except Exception as e:  # worker-level failure (model load, etc.)
        for spec in specs:
            q.put((spec.id, None, repr(e)))


def _score_parallel(specs, model_ref, secret_dir, ngpu, suite_name=""):
    """Round-robin the suite's benchmarks across `ngpu` GPUs and score them
    concurrently (one model replica per GPU)."""
    import multiprocessing as mp
    # The GPUs SLURM allocated us (CUDA_VISIBLE_DEVICES may be physical ids like
    # "2,5"); index into it so each worker gets a distinct allocated GPU.
    alloc = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    nworkers = min(ngpu, len(specs))
    buckets: list[list] = [[] for _ in range(nworkers)]
    for i, spec in enumerate(specs):
        buckets[i % nworkers].append(spec)
    print(f"  scoring {len(specs)} benchmarks across {nworkers} GPUs (parallel)", flush=True)
    ctx = mp.get_context("spawn")   # spawn: required for CUDA in subprocesses
    q = ctx.Queue()
    procs = []
    for g in range(nworkers):
        gpu = alloc[g] if g < len(alloc) else str(g)
        p = ctx.Process(target=_gpu_worker, args=(gpu, buckets[g], model_ref, secret_dir, q, suite_name))
        p.start()
        procs.append(p)
    _role = {s.id: s.role for s in specs}
    scores, errors = {}, []
    for _ in range(len(specs)):
        sid, sc, err = q.get()
        if err:
            errors.append((sid, err))
            print(f"  {sid:16s} FAILED: {err}", flush=True)
        else:
            scores[sid] = sc
            if _role.get(sid) == "held_out":      # never print held-out score to stdout
                print("  [a held_out benchmark — name, score, n all hidden from logs]", flush=True)
            else:
                print(f"  {sid:16s} mean={sc.mean:.4f} ci=[{sc.ci_low:.4f},{sc.ci_high:.4f}] n={sc.n}", flush=True)
    for p in procs:
        p.join()
    if errors:
        raise RuntimeError(f"{len(errors)} benchmark(s) failed: {errors}")
    return scores


def _resolve_ngpu() -> int:
    """GPUs to parallelize benchmarks over. EVAL_GPUS=<n> forces it; otherwise
    auto = however many GPUs the job was allocated (CUDA_VISIBLE_DEVICES). We then
    cap to the benchmark count in run(), so the allocation drives parallelism and
    nothing is hardcoded — a 5-benchmark suite uses 5, a 6-benchmark suite 6."""
    env = os.getenv("EVAL_GPUS", "").strip().lower()
    if env.isdigit():
        return int(env)
    cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    return len(cvd) if cvd else 1


def run(suite_path: str, model_ref: str, secret_dir: str = "", out: str = "scores.json",
        heldout_dir: str = "") -> dict[str, Any]:
    registry.discover()
    suite_name, specs = load_suite(suite_path)

    ngpu = _resolve_ngpu()
    if ngpu > 1 and len(specs) > 1 and not str(model_ref).startswith("stub:"):
        scores = _score_parallel(specs, model_ref, secret_dir, ngpu, suite_name)   # capped to len(specs) inside
    else:
        scores = _score_sequential(specs, model_ref, secret_dir, suite_name)

    composite = compute_composite(scores, specs)
    result = {
        "suite": suite_name,
        "model": model_ref,
        **composite.to_dict(),
    }
    # Behavioral fingerprint (deterministic greedy probe of the model). Lets share_finding DEDUP
    # behaviorally-identical submissions — a no-op intervention (e.g. a zero-valued ITI bias) or a
    # shared trained core that the eval correctly scores identically, but that were otherwise posted
    # as DISTINCT findings (polluting the forum + the idea-forecasting data). Additive + fail-safe:
    # a fingerprint error NEVER breaks scoring; old evals simply lack the field (dedup then no-ops).
    try:
        from aar.eval_pod.model_fingerprint import fingerprint_model_ref
        _fp = fingerprint_model_ref(model_ref)
        if _fp:
            result["model_fingerprint"] = _fp
            print(f"  model_fingerprint = {_fp}")
    except Exception as _e:   # noqa: BLE001
        print(f"  [fingerprint] skipped ({_e})", flush=True)
    # HELD-OUT ISOLATION (the boundary that actually enforces invisibility):
    # `--out` is the RESEARCH-readable handoff (the AAR side reads it, and could even
    # `cat` it). So we write the STRIPPED composite there — the held-out's score never
    # enters research-readable storage, and everything downstream (forum/leaderboard/
    # evaluate_model) inherits a held-out-free view. The FULL composite (incl the
    # held-out) is written ONLY to the eval-private held-out dir (the eval worker runs
    # as the secret eval user; we chmod 700 so the research user can't read it). The
    # human reads the generalization score from there, with eval credentials.
    from aar.benchmarks.composite import strip_held_out
    if composite.held_out:
        Path(out).write_text(json.dumps(strip_held_out(result), indent=2))
        hdir = heldout_dir or os.getenv("HELDOUT_SCORES_DIR", "")
        if hdir:
            Path(hdir).mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(hdir, 0o700)
            except OSError:
                pass
            (Path(hdir) / Path(out).name).write_text(json.dumps(result, indent=2))
            print(f"  wrote FULL incl held-out -> {Path(hdir) / Path(out).name} (eval-private, mode-700)")
        else:
            print("  WARN: held-out present but no --heldout-dir / HELDOUT_SCORES_DIR — full "
                  "held-out NOT persisted (research handoff IS stripped, so the AAR stays safe)",
                  flush=True)
        print(f"  wrote STRIPPED (no held-out) -> {out}")
    else:
        Path(out).write_text(json.dumps(result, indent=2))
    print("=" * 60)
    print(f"  HEADLINE {composite.headline * 100:+.2f}%   passes_filter={composite.passes_filter}")
    for name, frac in composite.closed.items():
        print(f"    closed[{name}] = {frac * 100:+.2f}%")
    for name, d in composite.filter_detail.items():
        thr = d.get("floor", d.get("baseline_ci_low"))
        gate = d.get("gate", "floor")
        print(f"    filter[{name}] = {d['mean']:.4f} ({gate} {thr}) -> {'PASS' if d['passed'] else 'FAIL'}")
    print(f"  wrote {out}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="path to suite YAML")
    ap.add_argument("--model", required=True, help="model ref (stub:perfect | stub:sycophantic | stub:weak | <path>)")
    ap.add_argument("--secret-dir", default="", help="dir holding benchmarks' secret data (toy benchmarks inline theirs)")
    ap.add_argument("--out", default="scores.json", help="RESEARCH-readable handoff (held-out STRIPPED)")
    ap.add_argument("--heldout-dir", default="",
                    help="EVAL-PRIVATE dir (mode-700) for the FULL scores incl held-out; "
                         "the human reads the generalization score here with eval credentials")
    args = ap.parse_args()
    run(args.suite, args.model, args.secret_dir, args.out, args.heldout_dir)


if __name__ == "__main__":
    main()
