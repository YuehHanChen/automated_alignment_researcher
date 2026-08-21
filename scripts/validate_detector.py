"""Validate the finetuned RAGTruth detector against the human-labeled test split — must
reproduce roughly the paper's ~0.80 response-level F1 before we adopt it as the ragtruth scorer.

Run on a GPU from the RAGTruth baseline/ dir (so test.jsonl is local), with PYTHONPATH=<repo>:
    RAGTRUTH_DETECTOR_BASE=meta-llama/Llama-2-13b-hf python validate_detector.py <adapter_dir> [N]

Gold = human label list non-empty (response-level). Reports overall + per-task P/R/F1 (no sklearn).
"""
from __future__ import annotations

import json
import random
import sys

from aar.benchmarks.ragtruth.detector import RagtruthDetector

adapter = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
rows = [json.loads(l) for l in open("test.jsonl")]
# stratify a subset across tasks for a bounded run (detector gen is 512 tokens/item, serialized)
random.Random(42).shuffle(rows)
rows = rows[:N]

det = RagtruthDetector(adapter)


def prf(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-9, p + r)


conf = {}          # task -> [tp, fp, tn, fn]
allc = [0, 0, 0, 0]
for i, rrow in enumerate(rows):
    tt = rrow["task_type"]
    gold = len(rrow.get("labels", [])) > 0
    pred = det.hallucinated(tt, rrow.get("question", ""), rrow.get("reference", ""), rrow["response"])
    c = conf.setdefault(tt, [0, 0, 0, 0])
    idx = (0 if gold and pred else 1 if (not gold) and pred else 2 if (not gold) and (not pred) else 3)
    c[idx] += 1
    allc[idx] += 1
    if (i + 1) % 50 == 0:
        print(f"  ...{i+1}/{len(rows)}", flush=True)

tp, fp, tn, fn = allc
p, r, f1 = prf(tp, fp, fn)
print(f"\n[DETECTOR vs human]  n={sum(allc)}")
print(f"  gold halluc rate={ (tp+fn)/max(1,sum(allc)):.3f}  pred halluc rate={(tp+fp)/max(1,sum(allc)):.3f}")
print(f"  OVERALL response-level  P={p:.3f} R={r:.3f} F1={f1:.3f}  acc={(tp+tn)/max(1,sum(allc)):.3f}  (tp{tp} fp{fp} tn{tn} fn{fn})")
for tt, c in sorted(conf.items()):
    p2, r2, f12 = prf(c[0], c[1], c[3])
    print(f"  {tt:9s}  P={p2:.3f} R={r2:.3f} F1={f12:.3f}  n={sum(c)}")
