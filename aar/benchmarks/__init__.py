"""Benchmark plugins for the multi-benchmark AAR harness.

A benchmark is a held-out scorer that lives entirely on the secret side (the
eval pod). It receives a *trained model* (never predictions) plus its own
secret data, runs generation / judging / rollout, and returns an aggregate
`BenchmarkScore`. The research side imports nothing from this package.

Three categories (see base.py): rule, judge, trajectory.
"""
