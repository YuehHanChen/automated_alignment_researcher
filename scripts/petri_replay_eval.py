"""Replay the Petri agentic-adversarial REGIME (system prompt + escalating user turns) to ONE model
(array task) and measure PER-TURN collapse. Comparing base_llama vs gwhspt under identical prompts
isolates the model from the regime:
  - gwhspt collapses, base stays coherent  -> DPO-induced, regime-triggered fragility (gwhspt-specific)
  - both collapse                          -> the regime itself drives it (not special to gwhspt)
Cross-check vs WildChat (benign chat, same models): gwhspt 18% / base 22% there.

Args: <replay.jsonl> <out_dir> <array_index>"""
import gc
import json
import statistics
import sys
from pathlib import Path

from aar.benchmarks.wildchat_degen.benchmark import H_FLOOR, L_FLOOR, _norm_entropy
from aar.eval_pod.models import load_model

MODELS = [
    ("base_llama", "meta-llama/Llama-3.2-3B-Instruct"),
    ("gwhspt", "/opt/aar/work/petri_retrain/power_seeking_gwhspt/model"),
]
DATA = sys.argv[1]
OUTDIR = Path(sys.argv[2]); OUTDIR.mkdir(parents=True, exist_ok=True)
idx = int(sys.argv[3])
tag, mp = MODELS[idx]
N, MAXTOK = 250, 768

items = [json.loads(l) for l in open(DATA) if l.strip()][:N]
print(f"=== [{idx}] {tag}: {mp} | {len(items)} replay items ===", flush=True)
model = load_model(mp)

# multi-turn rollout WITH the agentic system prompt, batched per turn-round (the multi_if idiom)
histories = [[{"role": "system", "content": it["system"]}] for it in items]
transcripts = [[] for _ in items]
maxturns = max(len(it["turns"]) for it in items)
for t in range(maxturns):
    active = [k for k, it in enumerate(items) if t < len(it["turns"])]
    prompts = [histories[k] + [{"role": "user", "content": items[k]["turns"][t]["prompt"]}] for k in active]
    outs = model.generate_batch(prompts, max_new_tokens=MAXTOK)
    for k, out in zip(active, outs):
        u = {"role": "user", "content": items[k]["turns"][t]["prompt"]}
        a = {"role": "assistant", "content": out}
        histories[k] += [u, a]; transcripts[k] += [u, a]


def feat(resp):
    w = resp.split()
    if len(w) < L_FLOOR:
        return True, len(w), 0.0          # terse/short collapse
    e = _norm_entropy(resp)
    return (e < H_FLOOR), len(w), e        # salad collapse iff entropy too low


per_turn = {}
for t in range(maxturns):
    coll = n = wsum = 0; elist = []
    for it, tr in zip(items, transcripts):
        ri = 2 * t + 1
        if t >= len(it["turns"]) or ri >= len(tr):
            continue
        c, w, e = feat(tr[ri]["content"]); coll += c; n += 1; wsum += w
        if w >= L_FLOOR:
            elist.append(e)
    per_turn[str(t + 1)] = {"n": n, "collapse_rate": round(coll / max(1, n), 4),
                            "mean_words": round(wsum / max(1, n), 1),
                            "mean_entropy": round(statistics.mean(elist), 4) if elist else 0.0}

res = {"tag": tag, "model": mp, "n": len(items), "per_turn": per_turn}
(OUTDIR / f"replay_{tag}.json").write_text(json.dumps(res, indent=2))
print("RESULT", json.dumps(res), flush=True)
try:
    del model
    import torch
    gc.collect(); torch.cuda.empty_cache()
except Exception:  # noqa: BLE001
    pass
print("DONE", flush=True)
