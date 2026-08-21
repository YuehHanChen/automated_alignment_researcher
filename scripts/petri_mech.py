"""Mechanism mining for the Petri<->WildChat degeneration gap. Pools every target (gwhspt) turn across
the t3+t5 audits and asks WHY/WHEN gwhspt collapses, separating the entangled drivers:

  1) DOSE-RESPONSE vs context length  -> collapse% binned by words-of-context-before-the-turn
  2) DOSE-RESPONSE vs turn depth      -> collapse% + mean entropy by target-turn index
  3) ONSET (pure length, pre-poison)  -> at the FIRST collapse, how much context had accrued, and was
                                         that context CLEAN (only good prior outputs)? Isolates a length
                                         threshold from the self-poisoning that follows.
  4) ABSORBING / SELF-POISON          -> P(collapse_{t+1} | collapse_t) vs P(collapse_{t+1} | clean_t);
                                         once collapsed, does it stay? (feedback-spiral test)
  5) SCAFFOLD                          -> size + snippet of the opening system/user context (is the 4000w
                                         an agentic tool scaffold, adversarial content, or both)
"""
import math
import re
import sys
from collections import Counter, defaultdict
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


def mtext(m):
    return (getattr(m, "text", None) or (m.content if isinstance(getattr(m, "content", None), str) else "") or "")


def target_turns(s):
    """[(resp_text, ctx_words_before, entropy, collapsed)] in order."""
    out, ctx = [], 0
    for m in (s.messages or []):
        t = mtext(m)
        if m.role == "tool" and "<target_response>" in t:
            resp = TR.sub("", t).split("Call send_message")[0].strip()
            w = resp.split()
            ent = norm_entropy(resp) if len(w) >= 2 else 0.0
            coll = (len(w) < L_FLOOR) or (ent < H_FLOOR)
            out.append((resp, ctx, ent, coll))
        ctx += len(t.split())
    return out


files = sys.argv[1:]
pts = []                       # (ctx_words, turn_idx, entropy, collapsed)
by_turn = defaultdict(list)    # turn_idx -> [collapsed]
ent_by_turn = defaultdict(list)
onset_ctx, onset_turn, onset_clean = [], [], 0
trans_clean = trans_collapsed = 0
# transition counts for self-poison test
c_given_c = n_given_c = c_given_clean = n_given_clean = 0
stay_collapsed = onset_then_all_collapsed = 0
n_onset = 0
scaffold_printed = False

for f in files:
    for s in (read_eval_log(f).samples or []):
        tt = target_turns(s)
        if not tt:
            continue
        if not scaffold_printed:
            first_user = next((mtext(m) for m in s.messages if m.role in ("system", "user")), "")
            print(f"=== SCAFFOLD (opening {len(first_user.split())} words) ===\n{first_user[:900]}\n=== end scaffold ===\n")
            scaffold_printed = True
        colls = [c for (_, _, _, c) in tt]
        for i, (_, ctx, ent, coll) in enumerate(tt):
            pts.append((ctx, i + 1, ent, coll))
            by_turn[i + 1].append(coll)
            ent_by_turn[i + 1].append(ent)
        # transitions
        for a, b in zip(colls, colls[1:]):
            if a:
                n_given_c += 1; c_given_c += b
            else:
                n_given_clean += 1; c_given_clean += b
        # onset
        if any(colls):
            k = colls.index(True)
            onset_ctx.append(tt[k][1]); onset_turn.append(k + 1)
            onset_clean += all(not colls[j] for j in range(k))  # all prior turns clean at onset (always true: k=first)
            n_onset += 1
            if all(colls[k:]):
                onset_then_all_collapsed += 1


def pct(a, b):
    return f"{100*a/b:.0f}%" if b else "n/a"

def med(x):
    x = sorted(x); return x[len(x)//2] if x else 0

print(f"pooled target turns: {len(pts)}  across {len(files)} audit file(s)\n")

print("1) COLLAPSE vs CONTEXT-LENGTH (words before the turn)")
bins = [(0,1000),(1000,2000),(2000,3000),(3000,4000),(4000,5000),(5000,99999)]
for lo,hi in bins:
    sub=[c for (ctx,_,_,c) in pts if lo<=ctx<hi]
    print(f"   ctx {lo:>5}-{hi if hi<99999 else 'inf':<5}: n={len(sub):>4}  collapse={pct(sum(sub),len(sub))}")

print("\n2) COLLAPSE + MEAN-ENTROPY by TURN INDEX")
for t in sorted(by_turn):
    cs=by_turn[t]; es=[e for e in ent_by_turn[t] if e>0]
    print(f"   turn {t}: n={len(cs):>4}  collapse={pct(sum(cs),len(cs)):>4}  mean_entropy={sum(es)/len(es):.3f}" if es else f"   turn {t}: n={len(cs)}")

print("\n3) ONSET of first collapse (context BEFORE self-poison)")
print(f"   transcripts that ever collapse: {n_onset}")
print(f"   median onset turn: {med(onset_turn)}  | median onset ctx-words: {med(onset_ctx)}")
print(f"   onset->stays-collapsed-thereafter: {pct(onset_then_all_collapsed, n_onset)}")

print("\n4) SELF-POISON / ABSORBING (next-turn collapse | this-turn state)")
print(f"   P(collapse_t+1 | collapse_t) = {pct(c_given_c, n_given_c)}   (n={n_given_c})")
print(f"   P(collapse_t+1 | clean_t)    = {pct(c_given_clean, n_given_clean)}   (n={n_given_clean})")
