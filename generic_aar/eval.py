"""Generic eval runner: register this folder's benchmarks, then score a model against a suite
using the shared harness engine (`aar.eval_pod.run_eval`) — same scoring/held-out/capability logic
as the 10-axis harness, no changes.

Usage (run from the repo root with PYTHONPATH=.):
  # no-GPU / no-key demo:
  python generic_aar/eval.py --suite generic_aar/suite.yaml --model stub:perfect \
      --heldout-dir generic_aar/_heldout
  # real model (baseline = your untrained HF id; or a trained checkpoint dir):
  python generic_aar/eval.py --suite generic_aar/suite.yaml --model <hf-id-or-path> \
      --secret-dir <published-items-dir> --out scores.json --heldout-dir generic_aar/_heldout

Output: `scores.json` = held-out-STRIPPED composite (headline geomean over the `safety` legs +
per-leg closed%, and the capability pass/fail). The full result incl. the held-out leg is written
to --heldout-dir (eval-private).
"""
from __future__ import annotations

import argparse

import generic_aar.benchmarks  # noqa: F401  (imports register the plugins before scoring)
from aar.eval_pod.run_eval import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a model on a custom AAR suite.")
    ap.add_argument("--suite", required=True, help="path to the suite YAML")
    ap.add_argument("--model", required=True,
                    help="HF id | local checkpoint dir | stub:perfect|weak|sycophantic")
    ap.add_argument("--secret-dir", default="",
                    help="dir with published <bench>.jsonl items (omit if benchmarks are inline)")
    ap.add_argument("--out", default="scores.json", help="research-readable output (held-out stripped)")
    ap.add_argument("--heldout-dir", default="", help="eval-private dir for the full held-out result")
    args = ap.parse_args()
    run(args.suite, args.model, args.secret_dir, args.out, args.heldout_dir)


if __name__ == "__main__":
    main()
