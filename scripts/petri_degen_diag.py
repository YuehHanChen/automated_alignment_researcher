"""Diagnose gwhspt's Petri collapse: is it triggered by adversarial CONTENT or by long CONTEXT/depth?

For each Petri audit transcript we extract the ORDERED sequence of target (gwhspt) responses (the
<target_response> tool messages) and, for the FIRST and LAST target turn, compute (a) normalized token
entropy + word count of the response and (b) the running context length (words of all prior messages).

If gwhspt already collapses on its FIRST response (short context) -> CONTENT-driven. If only the deep/late
responses collapse while early ones are clean -> CONTEXT-LENGTH/accumulation-driven. Also dumps a few
example collapsed responses + the adversarial auditor turns that preceded them.
"""
import math
import re
import sys
from collections import Counter
from inspect_ai.log import read_eval_log

H_FLOOR, L_FLOOR = 0.9, 50
TR = re.compile(r"</?target_response>|\[message_id: M\d+\]")


def norm_entropy(text):
    w = text.split()
    if len(w) < 2:
        return 0.0
    c = Counter(w); tot = len(w)
    H = -sum((v / tot) * math.log2(v / tot) for v in c.values())
    return H / math.log2(len(c)) if len(c) > 1 else 0.0


def msg_text(m):
    t = m.text if hasattr(m, "text") and m.text else (m.content if isinstance(getattr(m, "content", None), str) else "")
    return t or ""


def target_turns(s):
    """Ordered list of (response_text, context_words_before_it) for each target response in the audit."""
    out, ctx_words = [], 0
    for m in (s.messages or []):
        t = msg_text(m)
        if m.role == "tool" and "<target_response>" in t:
            resp = TR.sub("", t).split("Call send_message")[0].strip()
            out.append((resp, ctx_words))
        ctx_words += len(t.split())
    return out


def collapsed(resp):
    w = resp.split()
    return (len(w) < L_FLOOR) or (norm_entropy(resp) < H_FLOOR)


ev = sys.argv[1]
log = read_eval_log(ev)
samples = log.samples or []
first_coll = last_coll = both = neither = 0
first_ctx, last_ctx = [], []
n_with_turns = 0
examples = []
for s in samples:
    tt = target_turns(s)
    if not tt:
        continue
    n_with_turns += 1
    (r0, c0), (rl, cl) = tt[0], tt[-1]
    fc, lc = collapsed(r0), collapsed(rl)
    first_coll += fc; last_coll += lc
    both += (fc and lc); neither += (not fc and not lc)
    first_ctx.append(c0); last_ctx.append(cl)
    if len(examples) < 4 and lc:
        examples.append((len(tt), c0, r0[:280], cl, rl[:280]))

def med(x):
    x = sorted(x); return x[len(x) // 2] if x else 0

print(f"file: {ev.split('/')[-1]}")
print(f"transcripts with >=1 target turn: {n_with_turns}/{len(samples)}")
print(f"avg target turns/transcript: {sum(1 for s in samples for _ in target_turns(s))/max(1,n_with_turns):.1f}")
print(f"FIRST-response collapse rate : {first_coll}/{n_with_turns} = {100*first_coll/max(1,n_with_turns):.0f}%   (ctx-words median {med(first_ctx)})")
print(f"LAST-response  collapse rate : {last_coll}/{n_with_turns} = {100*last_coll/max(1,n_with_turns):.0f}%   (ctx-words median {med(last_ctx)})")
print(f"both collapse: {both}  | neither: {neither}")
print("=== examples (n_turns | first ctx-words | first resp[:280] || last ctx-words | last resp[:280]) ===")
for nt, c0, r0, cl, rl in examples:
    print(f"\n[turns={nt}] FIRST ctx={c0}w :: {r0!r}")
    print(f"            LAST  ctx={cl}w :: {rl!r}")
