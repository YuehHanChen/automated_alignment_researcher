"""Validate local-judge reliability for the two hallucination (b) legs against GROUND TRUTH,
before trusting their baselines (run on a GPU — loads the local Qwen2.5-7B judge):

    JUDGE_MODEL_LOCAL=Qwen/Qwen2.5-7B-Instruct python scripts/validate_judges.py

(1) RAGTRUTH — the corpus ships HUMAN hallucination annotations. Run the ragtruth faithfulness
    judge over annotated TEST responses; gold = "hallucinated" iff the response has >=1 annotated
    span. Reports hallucination-DETECTION precision/recall/F1 + accuracy + the judge's vs gold's
    hallucination RATE (the leniency check — if judge_rate << gold_rate, the judge under-detects).
(2) TRUTHFULQA-GEN — feed the truthful-judge each item's gold BEST answer (expect truthful=YES)
    and a gold INCORRECT answer (expect truthful=NO); reports discrimination accuracy on knowns.
"""
from __future__ import annotations

import json
import random
import urllib.request

from aar.eval_pod.judges import make_local_judge
from aar.benchmarks.ragtruth.benchmark import faithful_prompt
from aar.benchmarks.truthfulqa_gen.benchmark import truthful_prompt

N_RAG = 150
N_TQA = 120
judge = make_local_judge()   # JUDGE_MODEL_LOCAL (default Qwen2.5-7B-Instruct)


def fetch_jsonl(url: str):
    with urllib.request.urlopen(url, timeout=180) as r:
        return [json.loads(l) for l in r.read().decode("utf-8").splitlines() if l.strip()]


# ---- (1) RAGTruth: judge vs human labels --------------------------------------------------
base = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/"
resp = fetch_jsonl(base + "response.jsonl")
si = {d["source_id"]: d for d in fetch_jsonl(base + "source_info.jsonl")}
pool = [r for r in resp if r.get("split") == "test"
        and si.get(r["source_id"], {}).get("task_type") in ("QA", "Summary")]
random.Random(42).shuffle(pool)
pool = pool[:N_RAG]
tp = fp = tn = fn = 0
for r in pool:
    rec = si[r["source_id"]]
    ctx = rec["source_info"]
    ctx = ctx if isinstance(ctx, str) else json.dumps(ctx, ensure_ascii=False)
    gold_halluc = len(r.get("labels") or []) > 0
    pred_halluc = not bool(judge(faithful_prompt(ctx, r["response"])))
    if gold_halluc and pred_halluc:
        tp += 1
    elif gold_halluc and not pred_halluc:
        fn += 1
    elif (not gold_halluc) and pred_halluc:
        fp += 1
    else:
        tn += 1
tot = tp + fp + tn + fn
acc = (tp + tn) / max(1, tot)
prec = tp / max(1, tp + fp)
rec = tp / max(1, tp + fn)
f1 = 2 * prec * rec / max(1e-9, prec + rec)
print(f"[RAGTRUTH judge vs human]  n={tot}")
print(f"  gold hallucination rate = {(tp + fn) / max(1, tot):.3f}   "
      f"judge hallucination rate = {(tp + fp) / max(1, tot):.3f}   (leniency: judge<<gold => under-detects)")
print(f"  accuracy={acc:.3f}   hallucination-detect  P={prec:.3f} R={rec:.3f} F1={f1:.3f}   "
      f"(tp={tp} fp={fp} tn={tn} fn={fn})")

# ---- (2) TruthfulQA-gen: truthful-judge discrimination on knowns ---------------------------
from datasets import load_dataset
ds = list(load_dataset("truthfulqa/truthful_qa", "generation", split="validation"))
random.Random(42).shuffle(ds)
ok = n = 0
for it in ds[:N_TQA]:
    corr = list(it["correct_answers"])
    incorr = list(it["incorrect_answers"])
    if not corr or not incorr:
        continue
    yes_on_correct = bool(judge(truthful_prompt(it["question"], corr[0], corr, incorr)))
    no_on_incorrect = not bool(judge(truthful_prompt(it["question"], incorr[0], corr, incorr)))
    ok += int(yes_on_correct) + int(no_on_incorrect)
    n += 2
print(f"[TRUTHFULQA-gen truthful-judge]  items={n // 2}  "
      f"discrimination_acc={ok / max(1, n):.3f}  (gold-correct->YES & gold-incorrect->NO)")
