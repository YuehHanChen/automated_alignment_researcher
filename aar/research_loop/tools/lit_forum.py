"""Shared FS **literature-review forum** — the sibling of ``fs_forum`` for the
literature survey instead of experimental findings.

A team's lit forum lives at ``LIT_FORUM_DIR`` (set per-team by the launcher, e.g.
``aar_litreview/<TEAM_ID>/``). It is **shared** exactly like the findings forum:
the literature-review pre-phase agents populate it (30+ method/paper entries
before the AARs start), and the team's AARs both **read** it (``get_literature``)
and **append** to it (``share_literature``) when their per-iteration search turns
up something useful. One atomic JSON file per entry; no server.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def _lit_dir() -> Path | None:
    """The team's OWN lit-forum dir (write target), or None if LIT_FORUM_DIR is
    unset/empty (so we NEVER fall back to Path('') == cwd and pollute the cwd).
    Under TEAM_DIR/litreview — entries the team's AARs add during THIS run, private
    to the team."""
    v = (os.getenv("LIT_FORUM_DIR") or "").strip()
    return Path(v) if v else None


def _axis_lit_dir() -> Path | None:
    """The AXIS-WISE literature baseline (read-only reference), or None. Set via
    LIT_AXIS_DIR (e.g. aar_litreview/<axis>/). Produced once per axis by the survey
    pre-phase and shared by every team on that axis. Teams READ it but never write
    here (their own in-run additions go to LIT_FORUM_DIR), so one team's discoveries
    stay invisible to other teams."""
    v = (os.getenv("LIT_AXIS_DIR") or "").strip()
    return Path(v) if v else None


# Fields a literature entry should carry (free-form text, but structured so the
# AARs and the dashboard can read it consistently).
_FIELDS = ("method", "category", "summary", "intuition", "core_mechanism",
           "reproduction_recipe", "prerequisites", "training_data", "evaluation",
           "key_results", "applicability", "source", "relevance", "by")


def write_lit_entry(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist one literature entry to the shared lit forum. uuid-named so many
    agents writing concurrently never collide."""
    d = _lit_dir()
    if d is None:
        return {"_saved_path": None, "error": "LIT_FORUM_DIR unset"}
    d.mkdir(parents=True, exist_ok=True)
    e = {k: payload.get(k) for k in _FIELDS}
    eid = uuid.uuid4().hex
    e["id"] = eid
    e["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(e.get("method") or "entry"))[:48]
    p = d / f"{eid}_{safe}.json"
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(e, f, indent=2, default=str)
    tmp.rename(p)
    e["_saved_path"] = str(p)
    return e


def read_lit_entries() -> list[dict[str, Any]]:
    """Load every literature entry visible to this team: the AXIS baseline survey
    (LIT_AXIS_DIR, read-only, shared across same-axis teams) PLUS the team's OWN
    in-run additions (LIT_FORUM_DIR). Axis entries first, then the team's; deduped
    by id (a team copy wins). A team never sees another team's in-run additions."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for d in (_axis_lit_dir(), _lit_dir()):           # axis baseline, then team-own
        if d is None or not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                e = json.loads(p.read_text())
            except Exception:
                continue
            eid = str(e.get("id") or p.stem)
            if eid in seen:
                continue
            seen.add(eid)
            out.append(e)
    return out


def count() -> int:
    n = 0
    for d in (_axis_lit_dir(), _lit_dir()):
        if d is not None and d.exists():
            n += len(list(d.glob("*.json")))
    return n
