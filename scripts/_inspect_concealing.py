import glob
from inspect_ai.log import read_eval_log

ev = sorted(glob.glob("/opt/aar/eval-user/petri_rerun_fixed/concealing_uncertainty/winner/mt6/*.eval"))[-1]
log = read_eval_log(ev)
print("eval status:", getattr(log, "status", "?"))
err = getattr(log, "error", None)
if err:
    print("LOG ERROR:", str(err)[:500])
s = (log.samples or [])[0]
print("sample.error:", str(getattr(s, "error", None))[:400])
# scan events for the target model call + any error text
evs = getattr(s, "events", None) or []
print("n events:", len(evs))
for e in evs:
    et = type(e).__name__
    txt = ""
    for attr in ("error", "message", "content", "output"):
        v = getattr(e, attr, None)
        if v:
            txt = str(v)[:200]; break
    if "error" in et.lower() or "Model" in et or txt and ("error" in txt.lower() or "Error" in txt):
        print(f"  [{et}] {txt!r}")
