"""Mechanism test: does DOUBLE-BOS cause the gwhspt collapse? Same static Petri prompts, HF generation,
two conditions:
  - single (mode=single): tokenizer.apply_chat_template (correct: ONE bos_token, exactly what HF replay did -> 8%)
  - double (mode=double): prepend ONE extra bos_token_id to every turn's input (mimics vLLM chat double-BOS)
If double ~= 83% (vLLM) and single ~= 8%, double-BOS is the proven root cause of the Petri "degeneration".

Args: <replay.jsonl> <out_json> <single|double> <n>"""
import json
import math
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/opt/aar/work/petri_retrain/power_seeking_gwhspt/model"
DATA, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 250
BS = 16
H_FLOOR, L_FLOOR = 0.9, 50

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
BOS = tok.bos_token_id
items = [json.loads(l) for l in open(DATA) if l.strip()][:N]
print(f"MODE={MODE}  BOS={BOS}  n={len(items)}", flush=True)


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


def render_ids(msgs):
    """Correct single-BOS token ids: render to a STRING (template emits one bos_token), then tokenize
    with add_special_tokens=False so NO second bos is added -> exactly the HFModel path that gave 8%."""
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    if isinstance(s, list):
        s = s[0]
    return tok(s, add_special_tokens=False).input_ids


_verified = False
histories = [[{"role": "system", "content": it["system"]}] for it in items]
transcripts = [[] for _ in items]
maxturns = max(len(it["turns"]) for it in items)
for t in range(maxturns):
    active = [k for k, it in enumerate(items) if t < len(it["turns"])]
    for b in range(0, len(active), BS):
        batch = active[b:b + BS]
        seqs = []
        for k in batch:
            msgs = histories[k] + [{"role": "user", "content": items[k]["turns"][t]["prompt"]}]
            base = render_ids(msgs)
            ids = ([BOS] + base) if MODE == "double" else base   # double = prepend ONE extra bos
            if not _verified:
                lead = sum(1 for x in ids[:4] if x == BOS)
                print(f"  [verify] mode={MODE} leading-BOS={lead} first6={ids[:6]}", flush=True)
                _verified = True
            seqs.append(torch.tensor(ids, dtype=torch.long))
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), maxlen), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, maxlen - len(s):] = s
            attn[i, maxlen - len(s):] = 1
        input_ids, attn = input_ids.cuda(), attn.cuda()
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=768,
                                 do_sample=True, temperature=0.6, top_p=0.9, pad_token_id=tok.pad_token_id)
        for i, k in enumerate(batch):
            gen = tok.decode(out[i, maxlen:], skip_special_tokens=True)
            histories[k] += [{"role": "user", "content": items[k]["turns"][t]["prompt"]},
                             {"role": "assistant", "content": gen}]
            transcripts[k] += [{"role": "user", "content": items[k]["turns"][t]["prompt"]},
                               {"role": "assistant", "content": gen}]
    print(f"  turn {t+1} done", flush=True)

per_turn = {}
for t in range(maxturns):
    coll = n = wsum = 0; el = []
    for it, tr in zip(items, transcripts):
        ri = 2 * t + 1
        if t >= len(it["turns"]) or ri >= len(tr):
            continue
        c, w, e = feat(tr[ri]["content"]); coll += c; n += 1; wsum += w
        if w >= L_FLOOR:
            el.append(e)
    per_turn[str(t + 1)] = {"n": n, "collapse_rate": round(coll / max(1, n), 4),
                            "mean_words": round(wsum / max(1, n), 1),
                            "mean_entropy": round(sum(el) / max(1, len(el)), 4) if el else 0.0}

res = {"mode": MODE, "engine": "hf", "n": len(items), "per_turn": per_turn}
open(OUT, "w").write(json.dumps(res, indent=2))
print("RESULT", json.dumps(res), flush=True)
for i, tr in enumerate([tr for tr in transcripts if len(tr) > 3][:2]):
    print(f"--- {MODE} turn-2 ex{i}[:250] ---\n{tr[3]['content'][:250]!r}", flush=True)
print("DONE", flush=True)
