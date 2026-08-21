"""Core types + category base classes for benchmark plugins.

The universal contract is **submit-model**: a benchmark is handed a model (an
object exposing `generate(prompt: str) -> str`) and its own secret directory,
and returns a `BenchmarkScore`. The three category base classes differ only in
*how* they turn a model into a score:

- RuleBenchmark       — generate outputs on held inputs, deterministic match vs held answers.
- JudgeBenchmark      — generate outputs, an LLM judge scores them vs a held reference/rubric.
- TrajectoryBenchmark — roll the model out in a held environment, grade the transcript.

A benchmark plugin lives in `benchmarks/<name>/` as a subclass that sets
`name` (class var) and implements the category's one abstract hook. Subclasses
auto-register via `__init_subclass__`, mirroring safety_aar's pattern.
"""
from __future__ import annotations

import concurrent.futures
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Concurrency for judge (API) calls within a benchmark. Judge calls are I/O-bound
# and thread-safe, so we fan them out. Env-configurable (JUDGE_CONCURRENCY) so an
# API judge (gpt-4o) can use the account's parallel budget — e.g. 32 per leg, which
# across a sharded suite (≤3 judge legs) stays ≤ ~100 concurrent. Default 12 (local).
# NB: each sharded GPU process runs its own pool, so total concurrent ≈ this × the
# number of judge legs running at once — size it to the API limit / shard fan-out.
JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "12"))
from typing import Any, Callable, Literal, Protocol

Category = Literal["rule", "judge", "trajectory"]
Role = Literal["safety", "capability_filter", "held_out"]


# ---------------------------------------------------------------------------
# Result + spec types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkScore:
    """Aggregate score for one benchmark. The ONLY thing that crosses the
    isolation boundary back to the research side — never individual items.

    `extra` (2026-06-21) = optional AGGREGATE sub-metrics that DECOMPOSE the headline so the AAR can
    see WHICH factor binds (e.g. privacylens's product P(helpful AND not-leak) -> {helpful_rate,
    leak_rate, no_action_rate}). Still aggregate-only (no per-item data crosses the boundary). Surfaced
    in scores.json's per_benchmark so evaluate_model returns it to the AAR each iteration."""
    mean: float
    ci_low: float
    ci_high: float
    n: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"mean": self.mean, "ci_low": self.ci_low, "ci_high": self.ci_high, "n": self.n}
        if self.extra:
            d["decomposition"] = self.extra
        return d


@dataclass(frozen=True)
class BenchmarkSpec:
    """How a suite entry parameterizes a benchmark. baseline/optimum are used
    by the composite (headroom-closed); floor gates capability_filter roles.

    `id` is the per-suite key (defaults to `name`). It lets the same benchmark
    appear under two entries — e.g. tracked as `safety` AND used as a
    `capability_filter` gate — without colliding."""
    name: str
    category: Category
    role: Role = "safety"
    id: str = ""
    baseline: float = 0.0
    optimum: float = 1.0
    floor: float | None = None          # for capability_filter: pass iff mean >= floor
    subset_size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", self.name)


class Model(Protocol):
    """Minimal model interface a benchmark depends on. A real model wraps an HF
    pipeline; the toy/stub model is just a Python callable. Keeping benchmarks
    coupled to this Protocol (not to torch) is what makes them CPU-toy-runnable."""
    def generate(self, prompt: str, **kwargs: Any) -> str: ...

    def candidate_logits(self, prompt: str, candidates: list[str],
                         use_chat_template: bool = True) -> list[float]:
        """First-token logit for each candidate continuation string after the
        prompt. Used by logprob-scored rule benchmarks (wei P(No), MMLU argmax)
        instead of free generation. Optional — generation-only models may omit
        it, in which case logprob benchmarks won't run against them."""
        ...

    def generate_batch(self, prompts: list[str], **kwargs: Any) -> list[str]:
        """Batched generation (the GPU throughput win). Returns one output per
        prompt, in order."""
        ...

    def candidate_logits_batch(self, prompts: list[str], candidates: list[str],
                               use_chat_template: bool = True) -> list[list[float]]:
        """Batched candidate_logits: one logit-list per prompt, in order."""
        ...

    def completion_logprob_batch(self, prompts: list[str], completions: list[str],
                                 use_chat_template: bool = False) -> list[float]:
        """LENGTH-NORMALIZED continuation log-prob (mean token log-prob) of
        `completions[i]` given `prompts[i]` — one float per index. Used by FACTOR
        (argmax over a prefix's 4 completions = the factual one), matching AI21's
        eval_factuality.py (sum token NLL over the completion span / completion len).
        Distinct from candidate_logits_batch, which returns the SUMMED (un-normalized)
        loglik that TruthfulQA-MC2's lm-eval scoring needs."""
        ...


def bootstrap_ci(values: list[float], n_resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for the mean. Ported from safety_aar/eval_server/common.py."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples)]


def score_from_judgments(judgments: list[float]) -> BenchmarkScore:
    """Build a BenchmarkScore from a list of per-item 0/1 (or continuous) judgments.

    Defensively DROPS any None (a judge SKIP) so a caller that forwards raw verdicts
    can't accidentally count a skip as 0 — the board-wide rule (judges.py) is that an
    unjudgeable item is excluded, never defaulted to an answer."""
    judgments = [j for j in judgments if j is not None]
    if not judgments:
        return BenchmarkScore(0.0, 0.0, 0.0, 0)
    mean = sum(judgments) / len(judgments)
    lo, hi = bootstrap_ci(judgments)
    return BenchmarkScore(mean=mean, ci_low=lo, ci_high=hi, n=len(judgments))


def score_from_verdicts(verdicts: list[Any]) -> BenchmarkScore:
    """Build a BenchmarkScore from per-item judge verdicts True/False/None.

    None = SKIP (the judge could not grade the item, e.g. the API kept erroring):
    the item is EXCLUDED — it counts toward neither the numerator nor the
    denominator, and `n` is the number of items ACTUALLY judged. This enforces the
    board-wide rule that a judge error is never defaulted to correct/incorrect.
    """
    n_total = len(verdicts)
    judged = [v for v in verdicts if v is not None]
    if len(judged) < n_total:
        import logging
        logging.getLogger(__name__).warning(
            "judge SKIPPED %d/%d items (unjudgeable after retries) — excluded from the metric",
            n_total - len(judged), n_total)
    return score_from_judgments([1.0 if v else 0.0 for v in judged])


# ---------------------------------------------------------------------------
# Registry (auto-populated via __init_subclass__)
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type["Benchmark"]] = {}


def register(cls: type["Benchmark"]) -> None:
    name = getattr(cls, "name", None)
    if not name:
        return  # abstract base / category class with no name — skip
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Duplicate benchmark name {name!r}: {_REGISTRY[name].__qualname__} and {cls.__qualname__}")
    _REGISTRY[name] = cls


def get(name: str) -> type["Benchmark"]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_benchmarks() -> dict[str, type["Benchmark"]]:
    return dict(_REGISTRY)


# ---------------------------------------------------------------------------
# Benchmark base + 3 category bases
# ---------------------------------------------------------------------------
class Benchmark(ABC):
    """Base for all benchmark plugins. Set `name` and `category` as class vars."""
    name: str = ""
    category: Category

    def __init__(self, spec: BenchmarkSpec, secret_dir: str):
        self.spec = spec
        self.secret_dir = secret_dir

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only register concrete, named plugins — not the 3 category bases.
        if getattr(cls, "name", ""):
            register(cls)

    @abstractmethod
    def score(self, model: Model) -> BenchmarkScore:
        """Run this benchmark against `model` using secret data. Returns aggregate only."""
        ...


class RuleBenchmark(Benchmark):
    """Deterministic scoring of model outputs against held gold answers."""
    category: Category = "rule"

    @abstractmethod
    def load_items(self) -> list[dict[str, Any]]:
        """Each item: {'prompt': str, 'answer': <gold>}. Loaded from secret_dir."""
        ...

    @abstractmethod
    def matches(self, output: str, gold: Any) -> bool:
        """Deterministic correctness check."""
        ...

    def score(self, model: Model) -> BenchmarkScore:
        items = self.load_items()
        if self.spec.subset_size:
            items = items[: self.spec.subset_size]
        prompts = [it["prompt"] for it in items]

        def _ok(o):   # a real generation = a non-empty string
            return o if (isinstance(o, str) and o.strip()) else None

        # Batched fast path; if the whole batch errors (e.g. OOM under load) fall through to per-item retry.
        try:
            outputs = [_ok(o) for o in model.generate_batch(prompts)]
        except Exception:
            outputs = [None] * len(prompts)
        # RETRY generations that FAILED or came back EMPTY (a serving error / empty completion is NOT a wrong
        # answer). Per-item retry (smaller footprint) recovers a transient OOM; a genuinely-empty deterministic
        # output stays empty and is SKIPPED below. Mirrors the audit's retry-timeouts + the judge SKIP rule.
        for _ in range(2):
            todo = [i for i, o in enumerate(outputs) if o is None]
            if not todo:
                break
            for i in todo:
                try:
                    outputs[i] = _ok(model.generate(prompts[i]))
                except Exception:
                    outputs[i] = None
        # SKIP items that never produced a real output (excluded from the metric, never scored wrong) —
        # the board-wide rule (see score_from_verdicts) applied to rule benchmarks.
        n_skipped = sum(1 for o in outputs if o is None)
        if n_skipped:
            import logging
            logging.getLogger(__name__).warning(
                "%s: %d/%d items had no valid generation after retries — SKIPPED (not scored wrong)",
                getattr(self, "name", type(self).__name__), n_skipped, len(items))
        judgments = [1.0 if self.matches(o, it["answer"]) else 0.0
                     for it, o in zip(items, outputs) if o is not None]
        return score_from_judgments(judgments)


class JudgeBenchmark(Benchmark):
    """An LLM judge scores model outputs against a held reference/rubric.

    Subclasses provide items + a per-item judge prompt; `judge_fn` is injected
    (real = gpt-4o over httpx; toy = a deterministic stub) so the category is
    testable without an API key."""
    category: Category = "judge"

    def __init__(self, spec: BenchmarkSpec, secret_dir: str,
                 judge_fn: Callable[[str], bool] | None = None):
        super().__init__(spec, secret_dir)
        self.judge_fn = judge_fn or self._default_judge

    @abstractmethod
    def load_items(self) -> list[dict[str, Any]]:
        """Each item: {'prompt': str, 'reference': <ref/rubric>}."""
        ...

    @abstractmethod
    def judge_prompt(self, item: dict[str, Any], output: str) -> str:
        """Render the grading prompt fed to the judge."""
        ...

    def _default_judge(self, judge_prompt: str) -> bool:
        raise RuntimeError(
            f"{type(self).__name__} needs a judge_fn (real gpt-4o or a stub). "
            "None was injected and there is no default."
        )

    def score(self, model: Model) -> BenchmarkScore:
        items = self.load_items()
        if self.spec.subset_size:
            items = items[: self.spec.subset_size]
        # Phase 1 (GPU): generate all outputs in a single batched call.
        outputs = model.generate_batch([it["prompt"] for it in items])
        # Phase 2 (API, parallel): judge them concurrently.
        prompts = [self.judge_prompt(it, o) for it, o in zip(items, outputs)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as ex:
            verdicts = list(ex.map(self.judge_fn, prompts))
        # verdicts are True/False/None — None (judge SKIP) is excluded from the metric.
        return score_from_verdicts(verdicts)


class TrajectoryBenchmark(Benchmark):
    """The model interacts with a held environment; grading is over the transcript.

    Server-side rollout: the benchmark owns the environment, drives the model
    turn-by-turn, and grades the resulting transcript. The model never sees the
    env definition or the grader."""
    category: Category = "trajectory"

    @abstractmethod
    def episodes(self) -> list[dict[str, Any]]:
        """Each episode is a seed/spec the environment uses to start a rollout."""
        ...

    @abstractmethod
    def rollout(self, model: Model, episode: dict[str, Any]) -> list[dict[str, str]]:
        """Drive the model through the env; return the transcript (list of turns)."""
        ...

    def rollout_batch(self, model: Model, episodes: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
        """Roll out many episodes. Default loops `rollout`; override to BATCH the
        per-turn generation across episodes (big GPU win — see SycEval)."""
        return [self.rollout(model, ep) for ep in episodes]

    @abstractmethod
    def grade(self, episode: dict[str, Any], transcript: list[dict[str, str]]) -> float:
        """Grade one transcript in [0,1]."""
        ...

    def score(self, model: Model) -> BenchmarkScore:
        eps = self.episodes()
        if self.spec.subset_size:
            eps = eps[: self.spec.subset_size]
        # Phase 1 (GPU): roll out every episode (batched per-turn if overridden).
        transcripts = self.rollout_batch(model, eps)
        # Phase 2 (judge/CPU, parallel): grade transcripts concurrently. grade()
        # may return None to EXCLUDE an episode (e.g. not initially correct).
        with concurrent.futures.ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as ex:
            grades = list(ex.map(lambda et: self.grade(et[0], et[1]), zip(eps, transcripts)))
        return score_from_judgments([g for g in grades if g is not None])
