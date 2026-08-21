"""Does vLLM's chat endpoint double-add BOS? Compare token IDs from vLLM's /tokenize (server ROOT, not
/v1) for a chat request vs HF apply_chat_template, for a few Petri conversations.

Args: <base_url ending in /v1> <replay.jsonl>"""
import json
import sys

import requests
from transformers import AutoTokenizer

BASE, DATA = sys.argv[1], sys.argv[2]
ROOT = BASE.rsplit("/v1", 1)[0]                  # /tokenize lives at the server root
MODEL = "/opt/aar/work/petri_retrain/power_seeking_gwhspt/model"
tok = AutoTokenizer.from_pretrained(MODEL)
BOS = tok.bos_token_id
items = [json.loads(l) for l in open(DATA) if l.strip()][:3]
print(f"BOS token id = {BOS} ({tok.decode([BOS])!r})\n")
for j, it in enumerate(items):
    msgs = [{"role": "system", "content": it["system"]},
            {"role": "user", "content": it["turns"][0]["prompt"]}]
    vids = []
    for url in (f"{ROOT}/tokenize", f"{BASE}/tokenize"):
        try:
            r = requests.post(url, json={"model": "target", "messages": msgs,
                                         "add_generation_prompt": True, "add_special_tokens": True}, timeout=120)
            jd = r.json()
            if j == 0:
                print(f"  [{url}] status={r.status_code} keys={list(jd)[:6] if isinstance(jd, dict) else type(jd)}")
            vids = jd.get("tokens") or jd.get("token_ids") or jd.get("input_ids") or []
            if vids:
                break
        except Exception as e:  # noqa: BLE001
            print(f"  [{url}] ERR {repr(e)[:80]}")
    hids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    hids = tok(hids if isinstance(hids, str) else hids[0], add_special_tokens=False).input_ids
    print(f"=== conversation {j} ===")
    print(f"  HF   first6={hids[:6]}  leadingBOS={sum(1 for x in hids[:4] if x == BOS)}  len={len(hids)}")
    print(f"  vLLM first6={vids[:6]}  leadingBOS={sum(1 for x in vids[:4] if x == BOS)}  len={len(vids)}")
    print(f"  len(vLLM)-len(HF) = {len(vids) - len(hids)}")
print("DONE")
