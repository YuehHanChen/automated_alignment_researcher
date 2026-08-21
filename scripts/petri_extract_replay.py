"""Extract the Petri TARGET REGIME (agentic system prompt + the auditor's escalating user turns) from the
gwhspt audits, so it can be replayed to base Llama AND gwhspt under identical conditions. This isolates
the MODEL from the REGIME: same agentic+adversarial prompts, different weights.

Output: one {"key", "system", "turns": [{"prompt": u1}, {"prompt": u2}, {"prompt": u3}]} per line."""
import json
import sys
from inspect_ai.log import read_eval_log


def extract(s):
    sys_msg, users = None, []
    for m in (s.messages or []):
        for tc in (getattr(m, "tool_calls", None) or []):
            a = tc.arguments or {}
            if tc.function == "set_system_message":
                sys_msg = str(a.get("system_message", "")).strip()
            elif tc.function == "send_message":
                txt = a.get("message", a.get("content", a.get("text", "")))
                if txt:
                    users.append(str(txt).strip())
    return sys_msg, users


files = sys.argv[1:-1]
out_path = sys.argv[-1]
rows, seen = [], set()
argkeys = set()
for f in files:
    for s in (read_eval_log(f).samples or []):
        # record send_message arg keys once for sanity
        for m in (s.messages or []):
            for tc in (getattr(m, "tool_calls", None) or []):
                if tc.function == "send_message":
                    argkeys |= set((tc.arguments or {}).keys())
        sys_msg, users = extract(s)
        if not sys_msg or len(users) < 2:
            continue
        users = users[:3]                      # collapse happens by turn 2-3; 3 turns suffices
        key = (sys_msg[:80], users[0][:80])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"key": f"petri_{len(rows)}", "system": sys_msg,
                     "turns": [{"prompt": u} for u in users]})

with open(out_path, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

import statistics
print("send_message arg keys seen:", sorted(argkeys))
print(f"wrote {len(rows)} replay items -> {out_path}")
print("turns/item:", sorted({len(r['turns']) for r in rows}))
print("system words: median", int(statistics.median(len(r["system"].split()) for r in rows)))
print("user-turn words: median", int(statistics.median(len(t["prompt"].split()) for r in rows for t in r["turns"])))
print("\n--- example item ---")
ex = rows[0]
print("SYSTEM[:300]:", ex["system"][:300])
for i, t in enumerate(ex["turns"]):
    print(f"USER{i+1}[:200]:", t["prompt"][:200])
