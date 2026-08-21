"""Publisher for privacylens — builds secret_dir/privacylens.jsonl from the PrivacyLens action-based
data (SALT-NLP/PrivacyLens `data/main_data.json`). The verbatim ToolEmu agent prompts are PRE-RENDERED
by `_render.py` in the isolated pl_venv (the source's procoder/toolemu deps can't go in the harness
venv); this publisher consumes that rendered artifact, caps to ≤300 (README rule 7, frozen seed), and
ships exactly the fields the plugin's two judges need.

The rendered artifact is produced once (see `_render.py` for the pl_venv setup):
  PYTHONPATH=<repo>/data_construction <PLV>/bin/python aar/benchmarks/privacylens/_render.py <repo> <rendered.json>

Item schema (one per line):
  {"prompt": <verbatim agent prompt>, "secrets": [str,...], "user_name": str,
   "user_instruction": str, "executable_trajectory": str}

Usage (standalone): python -m aar.benchmarks.privacylens._publish <rendered.json> <out.jsonl> [--n 300] [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def publish_privacylens(rendered_path: str, out_path: str, n: int = 300, seed: int = 42) -> dict:
    rendered = json.loads(Path(rendered_path).read_text())
    # ≤300 (rule 7): a frozen seeded sample (PrivacyLens items are homogeneous — all negative-norm
    # scenarios — so a uniform sample is representative; no stratification axis to preserve).
    pool = sorted(rendered, key=lambda r: r["name"])     # stable order before the seeded shuffle
    random.Random(seed).shuffle(pool)
    chosen = pool[:n]
    rows = [{"prompt": r["prompt"], "secrets": r["secrets"], "user_name": r["user_name"],
             "user_instruction": r["user_instruction"], "executable_trajectory": r["executable_trajectory"]}
            for r in chosen]
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"total": len(rows), "available": len(rendered),
            "secrets_per_item_min_max": (min(len(r["secrets"]) for r in rows),
                                         max(len(r["secrets"]) for r in rows))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("rendered")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(json.dumps(publish_privacylens(a.rendered, a.out, a.n, a.seed), indent=1))
