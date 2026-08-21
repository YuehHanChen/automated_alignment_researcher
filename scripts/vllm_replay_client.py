"""Replay the static Petri prompts through a vLLM OpenAI endpoint (served EXACTLY as Petri served the
target). Same prompts the HF replay used (HF gave gwhspt 2% collapse). If vLLM collapses (~86%) on these
identical prompts, the Petri degeneration is a SERVING artifact (engine-level, e.g. double-BOS), not a
model property and not the live auditor. If vLLM stays ~2%, the live adaptive auditor is the cause.

Args: <replay.jsonl> <out_json> <n> <base_url>"""
import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

DATA, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 250
BASE_URL = sys.argv[4] if len(sys.argv) > 4 else "http://localhost:8000/v1"
TEMP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
H_FLOOR, L_FLOOR = 0.9, 50

client = OpenAI(base_url=BASE_URL, api_key="dummy", timeout=600)
items = [json.loads(l) for l in open(DATA) if l.strip()][:N]


def norm_entropy(t):
    w = t.split()
    if len(w) < 2:
        return 0.0
    c = Counter(w); tot = len(w)
    H = -sum((v / tot) * math.log2(v / tot) for v in c.values())
    return H / math.log2(len(c)) if len(c) > 1 else 0.0


def feat(r):
    w = r.split()
    if len(w) < L_FLOOR:
        return True, len(w), 0.0
    e = norm_entropy(r)
    return (e < H_FLOOR), len(w), e


def run_item(it):
    msgs = [{"role": "system", "content": it["system"]}]
    resps = []
    for t in it["turns"]:
        msgs.append({"role": "user", "content": t["prompt"]})
        try:
            r = client.chat.completions.create(model="target", messages=msgs,
                                               temperature=TEMP, top_p=0.9, max_tokens=768, seed=0)
            out = r.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            out = f"[ERR {repr(e)[:80]}]"
        msgs.append({"role": "assistant", "content": out})
        resps.append(out)
    return resps


with ThreadPoolExecutor(max_workers=24) as ex:
    all_resps = list(ex.map(run_item, items))

maxturns = max(len(it["turns"]) for it in items)
per_turn = {}
for t in range(maxturns):
    coll = n = wsum = 0; el = []
    for resps in all_resps:
        if t >= len(resps):
            continue
        c, w, e = feat(resps[t]); coll += c; n += 1; wsum += w
        if w >= L_FLOOR:
            el.append(e)
    per_turn[str(t + 1)] = {"n": n, "collapse_rate": round(coll / max(1, n), 4),
                            "mean_words": round(wsum / max(1, n), 1),
                            "mean_entropy": round(sum(el) / max(1, len(el)), 4) if el else 0.0}

res = {"engine": "vllm", "tag": "gwhspt", "n": len(items), "decoding": f"temp{TEMP}_top_p0.9", "per_turn": per_turn}
open(OUT, "w").write(json.dumps(res, indent=2))
# also dump a few raw turn-2 responses so we can SEE whether vLLM produces salad
ex2 = [resps[1] for resps in all_resps if len(resps) > 1][:3]
print("RESULT", json.dumps(res), flush=True)
for i, r in enumerate(ex2):
    print(f"--- vLLM turn-2 example {i} [:300] ---\n{r[:300]!r}", flush=True)
print("DONE", flush=True)
