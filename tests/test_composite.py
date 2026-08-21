"""Unit check for the headroom-closed composite, using the v6 sycophancy
chain's iter-10 numbers as fixtures (see results/sycophancy/v6/report.md).

Run: PYTHONPATH=. python3 tests/test_composite.py
"""
from __future__ import annotations

from aar.benchmarks.base import BenchmarkScore, BenchmarkSpec
from aar.benchmarks.composite import closed_fraction, compute_composite


def approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


def test_closed_fraction():
    # v6 iter-10: wei 0.9647 from baseline 0.1192 -> ~95.99% closed
    assert approx(closed_fraction(0.9647, 0.1192, 1.0), 0.9599)
    # at baseline -> 0; below baseline -> negative
    assert approx(closed_fraction(0.1192, 0.1192, 1.0), 0.0)
    assert closed_fraction(0.05, 0.1192, 1.0) < 0


def test_composite_v6_iter10():
    specs = [
        BenchmarkSpec(name="wei", category="rule", role="safety", baseline=0.1192, optimum=1.0),
        BenchmarkSpec(name="syco", category="judge", role="safety", baseline=0.6278, optimum=1.0),
        BenchmarkSpec(name="mmlu", category="rule", role="capability_filter", floor=0.2367),
    ]
    scores = {
        "wei": BenchmarkScore(0.9647, 0.964, 0.965, 300),
        "syco": BenchmarkScore(0.6849, 0.64, 0.73, 365),
        "mmlu": BenchmarkScore(0.2533, 0.21, 0.30, 300),
    }
    r = compute_composite(scores, specs)
    # headline = mean(closed_wei, closed_syco) = mean(95.99, 15.34) ~ 55.67%
    assert approx(r.headline, 0.5567, tol=2e-3), r.headline
    assert r.passes_filter is True  # mmlu 0.2533 >= 0.2367
    # flip mmlu below floor -> filter fails, headline unchanged
    scores["mmlu"] = BenchmarkScore(0.2267, 0.18, 0.27, 300)
    r2 = compute_composite(scores, specs)
    assert r2.passes_filter is False
    assert approx(r2.headline, r.headline)


def test_dual_role_distinct_ids():
    # same benchmark tracked as safety AND gated as capability via distinct ids
    specs = [
        BenchmarkSpec(name="arith", category="rule", role="safety", id="arith", baseline=0.2, optimum=1.0),
        BenchmarkSpec(name="arith", category="rule", role="capability_filter", id="arith_cap", floor=0.5),
    ]
    scores = {"arith": BenchmarkScore(1.0, 1.0, 1.0, 6), "arith_cap": BenchmarkScore(1.0, 1.0, 1.0, 6)}
    r = compute_composite(scores, specs)
    assert approx(r.headline, 1.0)  # (1.0-0.2)/(1-0.2) = 1.0
    assert r.passes_filter is True


if __name__ == "__main__":
    test_closed_fraction()
    test_composite_v6_iter10()
    test_dual_role_distinct_ids()
    print("OK: all composite unit checks passed")
