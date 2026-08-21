"""Reconstruct the TARGET's (gwhspt's) actual conversation from Petri audits and compute the true
target-side dose-response, so it's apples-to-apples with WildChat (~800-word turn-3 context).

Target convo = set_system_message (its agentic mission prompt) + interleaved [send_message (user turn),
<target_response> (assistant turn)]. We report: size of the agentic system prompt, size of each auditor
user turn, and collapse% vs the TARGET's real accumulated context words (not the auditor scaffold)."""
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


def target_convo(s):
    """Return (sys_words, [(user_words, resp_text, resp_words, ent, collapsed, ctx_before)])."""
    sys_words = 0
    pending_user = 0          # words of the most recent send_message not yet paired to a response
    turns = []
    ctx = 0                   # running target-side context words
    for m in (s.messages or []):
        for tc in (getattr(m, "tool_calls", None) or []):
            a = tc.arguments or {}
            if tc.function == "set_system_message":
                sw = len(str(a.get("system_message", "")).split())
                sys_words = sw; ctx += sw
            elif tc.function == "send_message":
                uw = len(str(a.get("message", a.get("content", ""))).split())
                pending_user = uw
        t = mtext(m)
        if m.role == "tool" and "<target_response>" in t:
            resp = TR.sub("", t).split("Call send_message")[0].strip()
            w = resp.split()
            ent = norm_entropy(resp) if len(w) >= 2 else 0.0
            coll = (len(w) < L_FLOOR) or (ent < H_FLOOR)
            ctx_before = ctx + pending_user        # sys + prior turns + this user turn
            turns.append((pending_user, len(w), ent, coll, ctx_before))
            ctx = ctx_before + len(w)
            pending_user = 0
    return sys_words, turns


files = sys.argv[1:]
sys_sizes, user_sizes = [], []
pts = []                       # (target_ctx_words, turn_idx, collapsed)
by_turn = defaultdict(list)
for f in files:
    for s in (read_eval_log(f).samples or []):
        sysw, turns = target_convo(s)
        if not turns:
            continue
        sys_sizes.append(sysw)
        for i, (uw, rw, ent, coll, ctxb) in enumerate(turns):
            user_sizes.append(uw)
            pts.append((ctxb, i + 1, coll))
            by_turn[i + 1].append((ctxb, coll))


def med(x):
    x = sorted(x); return x[len(x) // 2] if x else 0

def pct(a, b):
    return f"{100*a/b:.0f}%" if b else "n/a"

print(f"transcripts: {len(sys_sizes)}  pooled target turns: {len(pts)}")
print(f"TARGET agentic system-prompt words: median {med(sys_sizes)}  (min {min(sys_sizes)}, max {max(sys_sizes)})")
print(f"auditor user-turn (send_message) words: median {med(user_sizes)}  p90 {sorted(user_sizes)[int(0.9*len(user_sizes))]}")
print(f"\nWildChat reference: turn-3 target context ~800 words; gwhspt collapse there ~18%\n")

print("TRUE TARGET-CONTEXT dose-response (collapse vs target-side words before the turn):")
for lo, hi in [(0,800),(800,1500),(1500,2500),(2500,3500),(3500,5000),(5000,99999)]:
    sub = [c for (ctx, _, c) in pts if lo <= ctx < hi]
    print(f"   target-ctx {lo:>5}-{hi if hi<99999 else 'inf':<5}: n={len(sub):>4}  collapse={pct(sum(sub),len(sub))}")

print("\nby TURN INDEX (median target-ctx words, collapse%):")
for t in sorted(by_turn):
    rows = by_turn[t]
    print(f"   turn {t}: n={len(rows):>4}  median_target_ctx={med([c for c,_ in rows]):>5}w  collapse={pct(sum(c for _,c in rows),len(rows))}")
