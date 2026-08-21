"""Print [mean, 95%CI-halfwidth, n] for a corrected Petri winner .eval — figure-ready, NaN-filtered."""
import math
import statistics
import sys

from inspect_ai.log import read_eval_log

scores = []
for s in (read_eval_log(sys.argv[1]).samples or []):
    for _, obj in (getattr(s, "scores", None) or {}).items():
        v = getattr(obj, "value", None)
        if isinstance(v, dict):
            for vv in v.values():
                try:
                    f = float(vv)
                    if f == f:  # not NaN
                        scores.append(f)
                    break
                except (TypeError, ValueError):
                    continue
        elif isinstance(v, (int, float)) and v == v:
            scores.append(float(v))
n = len(scores)
m = statistics.mean(scores)
ci = 1.96 * statistics.pstdev(scores) / math.sqrt(n) if n > 1 else 0.0
print(f"[{round(m, 3)}, {round(ci, 3)}, {n}]")
