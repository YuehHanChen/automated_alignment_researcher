"""Live dashboard for an AAR run — a single self-contained Flask app that reads
the shared-FS forum + scores + `squeue` and serves one auto-refreshing page.

Reads only research-readable data (forum / scores / chain logs) — never the
mode-700 holdout, so running it has no isolation implications.

Run on the cluster login node, then SSH-tunnel to view in your browser:

    # on the cluster:
    PYTHONPATH=<repo> python -m aar.web_ui.live_dashboard --port 8765
    # on your laptop:
    ssh -L 8765:localhost:8765 <cluster>   # then open http://localhost:8765
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from aar import config
from aar.research_loop.tools import fs_forum, lit_forum

RUNS = Path(os.getenv("HARNESS_RUNS_DIR", config.HARNESS_RUNS_DIR))
# Scores live in the team's SCORES_DIR (under TEAM_DIR/scores when pinned), NOT
# necessarily RUNS/scores — read it from config so a per-team dashboard sees its
# own team's scored runs.
SCORES = Path(os.getenv("SCORES_DIR", config.SCORES_DIR))
LOGS = Path(os.getenv("CHAIN_LOGS_DIR", f"/opt/aar/work/{os.getenv('USER', 'user')}/exp/logs"))
SESSION_LOGS = Path(os.getenv("SESSION_LOGS_DIR", config.LOGS_DIR))  # full structured per-session transcripts
JOB_PREFIX = os.getenv("CHAIN_JOB_PREFIX", "aar-syco-")
MAX_ITERS = int(os.getenv("MAX_ITERATIONS", "100"))

# Per-team forums live under FORUM_ROOT/<team_id>/. Each launch group (team) has
# its own forum; show the CURRENT team (the most recently active one) by default,
# or a pinned one via LOCAL_FINDINGS_DIR. Resolved per-request so the dashboard
# follows a newly launched team automatically.
FORUM_ROOT = Path(os.getenv("FORUM_ROOT", f"/opt/aar/work/{os.getenv('USER', 'user')}/aar_forum"))


def _forum_dir() -> Path:
    pinned = os.getenv("LOCAL_FINDINGS_DIR")
    if pinned:
        return Path(pinned)
    subs = [p for p in FORUM_ROOT.glob("*") if p.is_dir()]
    if subs:
        return max(subs, key=lambda p: p.stat().st_mtime)  # current (latest) team
    return FORUM_ROOT


def _team_id() -> str:
    """The current team's id. NEW layout: LOCAL_FINDINGS_DIR = aar_teams/<TEAM_ID>/forum,
    so the id is the forum dir's PARENT name. LEGACY layout: aar_forum/<TEAM_ID>, so it's
    the forum dir's own name. AAR_TEAM_ID overrides (the dashboard launcher sets it)."""
    env = os.getenv("AAR_TEAM_ID")
    if env:
        return env
    d = _forum_dir()
    return d.parent.name if d.name == "forum" else d.name


def _job_prefix() -> str:
    """Chain job-name prefix for the CURRENT team. Chains are named aar-<suite>-<model>-<seed>
    and the team_id is <suite>-<model>[-<agenttag>]-<timestamp>[-<n>], so we strip the timestamp
    (and any collision-uniquifier -<n> suffix), keep ONLY the <suite>-<model> fields, and prepend
    'aar-' (e.g. refusal-gemma-20260604-211817 -> 'aar-refusal-gemma-'; refusal-phi-opus48-...
    -> 'aar-refusal-phi-'). The optional 3rd field is the AGENT-model tag (e.g. opus48) which the
    chain job-names DON'T carry — including it would make the glob match nothing ('no chain found').
    CHAIN_JOB_PREFIX overrides; falls back to the module default if the team_id carries no timestamp."""
    env = os.getenv("CHAIN_JOB_PREFIX")
    if env:
        return env
    team = _team_id()
    base = re.sub(r"-\d{8}-\d{6}(-\d+)?$", "", team)   # -> <suite>-<model>[-<agenttag>]
    if base and base != team:
        core = "-".join(base.split("-")[:2])           # first two fields only = <suite>-<model>
        return f"aar-{core}-"
    return JOB_PREFIX


# The team's LITERATURE forum is the sibling of its findings forum (aar_litreview/<team>);
# the librarian agents populate it before the AARs start (run_litreview.py / lit_forum).
LIT_ROOT = Path(os.getenv("LIT_FORUM_ROOT", str(FORUM_ROOT.parent / "aar_litreview")))


def _lit_dir() -> Path:
    pinned = os.getenv("LIT_FORUM_DIR")
    return Path(pinned) if pinned else (LIT_ROOT / _team_id())


def _lit_entries() -> list:
    """Every literature entry the librarian shared for the current team (oldest-first)."""
    os.environ["LIT_FORUM_DIR"] = str(_lit_dir())   # point lit_forum at the current team
    try:
        es = lit_forum.read_lit_entries()
    except Exception:
        es = []
    es.sort(key=lambda e: e.get("created_at", ""))
    return es


def _lit_agent() -> dict:
    """The literature-review librarian's status: its squeue state (the lit phase runs once,
    before the AARs) + the latest progress lines from its log."""
    job = {}
    try:
        out = subprocess.run(["squeue", "-u", os.getenv("USER", ""), "-h", "-o", "%i|%j|%T|%M"],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            p = line.split("|")
            if len(p) >= 4 and "litreview" in p[1].lower():
                job = {"jobid": p[0].strip(), "state": p[2].strip(), "elapsed": p[3].strip()}
                break
    except Exception:
        pass
    tail, raw = [], []
    # The litreview job writes its .out to the SBATCH default (/opt/aar/work
    # while LOGS may be pinned to a team's logs dir (CHAIN_LOGS_DIR) where chain .outs live — so
    # search BOTH, else a per-team dashboard shows the librarian RUNNING but an empty trajectory.
    _lit_dirs = {LOGS, Path(f"/opt/aar/work/{os.getenv('USER', 'user')}/exp/logs")}
    logs = sorted({p for d in _lit_dirs for p in d.glob("aar-litreview*.out")},
                  key=lambda p: p.stat().st_mtime)  # NEWEST, not alphabetical
    if job.get("jobid"):   # while running, pin to THIS job's log (never an older run / 'test' log)
        m = [p for p in logs if p.name.endswith(f"_{job['jobid']}.out")]
        if m:
            logs = m
    if logs:
        try:
            lines = logs[-1].read_text(errors="ignore").splitlines()
            tail = [l for l in lines if "[litreview]" in l][-60:]   # progress summary (inline status)
            raw = [l for l in lines if l.strip()][-200:]            # full trajectory tail (popup)
        except Exception:
            pass
    return {"state": job.get("state") or ("done" if logs else "—"),
            "elapsed": job.get("elapsed"), "progress": tail, "log": raw}


def _pinned_team_id() -> str:
    """The pinned team id, or '' if not pinned to a specific team."""
    t = _team_id()
    return "" if (not t or t == FORUM_ROOT.name) else t


def _out_in_pinned_team(p, team=None) -> bool:
    """Does this chain .out belong to the pinned team? Each chain prints '[aar] team=<id>' near
    the top. True when unpinned (team='') so the 'latest team' view is unchanged. This is the ONE
    guard that keeps a FRESH team that REUSES an <axis>-<model> job prefix from inheriting a PRIOR
    team's .out files (which showed as phantom iterations / extra chains)."""
    team = _pinned_team_id() if team is None else team
    if not team:
        return True
    try:
        return f"team={team}" in p.read_text(errors="ignore")[:8000]
    except Exception:
        return False


def _chain_out_logs(chain: str):
    """A chain's .out logs FOR THE PINNED TEAM, oldest->newest. A soft-restart (scancel + resubmit,
    SAME team) gives a new job id (new .out) while prior .out(s) remain, so we aggregate across all
    of them. But a FRESH team reusing the same prefix must NOT inherit the PRIOR team's .out (see
    _out_in_pinned_team). Fall back to all (e.g. team= not yet flushed) to avoid showing nothing."""
    outs = sorted(LOGS.glob(f"{_job_prefix()}{chain}_*.out"),
                  key=lambda p: (p.stat().st_mtime, p.name))
    team = _pinned_team_id()
    if not team:
        return outs
    return [p for p in outs if _out_in_pinned_team(p, team)] or outs


def _chain_out_log(chain: str):
    outs = _chain_out_logs(chain)
    return outs[-1] if outs else None


def _chain_sessions(chain: str) -> list:
    """The session IDs belonging to a chain, in order — read from its .out log
    (each iteration prints '[Session N] session_<count>_<ts>'). This is how we
    attribute the shared-dir session transcripts back to a specific chain."""
    seen, out = set(), []
    for lg in _chain_out_logs(chain):  # oldest -> newest: cumulative across restarts
        try:
            text = lg.read_text(errors="ignore")
        except Exception:
            continue
        # session id may carry a chain tag (session_<n>_<chain>_<date>_<time>) — grab
        # the whole non-space token so chain-tagged AND legacy ids both attribute.
        for m in re.finditer(r"\[Session (\d+)\]\s+(session_\S+)", text):
            sid = m.group(2)
            if sid in seen:
                continue
            seen.add(sid)
            sp = SESSION_LOGS / f"{sid}.log"
            # index = lifetime iteration number (not the per-process one, which
            # resets to 0 on a soft-restart).
            out.append({"id": sid, "index": len(out),
                        "n_events": _count_events(sp), "exists": sp.exists()})
    return out


def _count_events(p: Path) -> int:
    if not p.exists():
        return 0
    try:
        return sum(1 for ln in p.read_text(errors="ignore").splitlines()
                   if re.match(r"^\[\d\d:\d\d:\d\d\] \w+$", ln))
    except Exception:
        return 0


def _session_events(session_id: str) -> list:
    """Parse one session transcript into ordered events (full reasoning + tool
    calls + inputs + results), each {ts, type, body}."""
    if not re.fullmatch(r"session_[\w.\-]+", session_id or ""):  # no '/': safe as a filename
        return []
    p = SESSION_LOGS / f"{session_id}.log"
    if not p.exists():
        return []
    events, cur = [], None
    for line in p.read_text(errors="ignore").splitlines():
        m = re.match(r"^\[(\d\d:\d\d:\d\d)\] (\w+)$", line)
        if m:
            if cur:
                events.append(cur)
            cur = {"ts": m.group(1), "type": m.group(2), "body": []}
        elif cur is not None and not line.startswith("# "):
            cur["body"].append(line)
    if cur:
        events.append(cur)
    for e in events:
        e["body"] = "\n".join(e["body"]).strip()[:8000]
    return events


def _squeue_jobs() -> dict:
    try:
        out = subprocess.run(
            ["squeue", "-u", os.getenv("USER", ""), "-h", "-o", "%i|%j|%T|%M"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        out = ""
    jobs = {}
    for line in out.strip().splitlines():
        p = line.split("|")
        if len(p) >= 4 and _job_prefix() in p[1]:
            jobs[p[1].strip()] = {"jobid": p[0].strip(), "state": p[2].strip(), "elapsed": p[3].strip()}
    return jobs


def _chain_status() -> list:
    jp = _job_prefix()
    jobs = _squeue_jobs()
    rows, seen = [], set()

    def _from_log(name):
        """iters / methods / shares / last-line, AGGREGATED across ALL the chain's
        .out logs so a soft-restart's counts stay CUMULATIVE (0s if no log yet).
        iters = DISTINCT session STARTS ('[Session N] session_<id>') — NOT
        count('[Session '), which double-counts (each session also prints a
        '[Session N] Completed' line when it finishes)."""
        logs = _chain_out_logs(name)
        if not logs:
            return 0, 0, 0, ""
        sids = set()
        me = sh = 0
        last = ""
        for lg in logs:  # oldest -> newest, so `last` settles on the newest log
            try:
                text = lg.read_text(errors="ignore")
            except Exception:
                continue
            for m in re.finditer(r"\[Session \d+\]\s+(session_\S+)", text):
                sids.add(m.group(1))
            me += text.count("__evaluate_model")
            sh += text.count("__share_finding")
            for line in reversed(text.splitlines()):
                if line.strip():
                    last = re.sub(r"^\[[0-9: ]*\]\s*|\[autonomous[^\]]*\]\s*", "", line).strip()[:140]
                    break
        return len(sids), me, sh, last

    # 1) Every chain in squeue — so a just-spawned chain SHOWS even before its .out
    #    log exists (the log can lag the job by minutes during a heavy first iteration).
    for jobname, j in sorted(jobs.items()):
        name = jobname.replace(jp, "")
        it, me, sh, last = _from_log(name)
        rows.append({"name": name, **j, "iterations": it, "methods": me, "shares": sh,
                     "last": last or ("(starting — no output yet)" if j.get("state") == "RUNNING" else ""),
                     "max_iters": MAX_ITERS})
        seen.add(name)

    # 2) Finished chains (no longer in squeue) that still have a log on disk.
    for lg in sorted(LOGS.glob(f"{jp}*.out")):
        if not _out_in_pinned_team(lg):
            continue
        m = re.match(rf"({re.escape(jp)}[^_]+)_(\d+)\.out", lg.name)
        if not m or m.group(1).replace(jp, "") in seen:
            continue
        name = m.group(1).replace(jp, "")
        it, me, sh, last = _from_log(name)
        rows.append({"name": name, "jobid": m.group(2), "state": "done", "elapsed": "",
                     "iterations": it, "methods": me, "shares": sh, "last": last, "max_iters": MAX_ITERS})
        seen.add(name)

    rows.sort(key=lambda r: r["name"])
    return rows


def _scores() -> dict:
    out, sd = {}, SCORES
    if sd.exists():
        for f in sorted(sd.glob("*.json")):
            try:
                out[f.stem] = json.loads(f.read_text())
            except Exception:
                pass
    return out


_CODE_TEXT_EXTS = (".py", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml",
                   ".cfg", ".toml", ".sh", ".ini")
_CODE_MAX_INLINE_BYTES = 60_000


BASELINES_PATH = Path(os.getenv("BASELINES_PATH", str(RUNS / "baselines.json")))


def _baselines() -> dict:
    """Per-benchmark base-model scores + floors, for showing raw scores alongside
    the baseline in the forum/leaderboard. NOT secret (the base model's own public
    scores) — written by the publisher on (re)publish, keyed by benchmark name:
    {"gsm8k": {"baseline": 0.86, "floor": 0.81}, "sycophancy_eval": {"baseline": 0.46}, ...}."""
    try:
        if BASELINES_PATH.exists():
            return json.loads(BASELINES_PATH.read_text())
    except Exception:
        pass
    return {}


def _enrich_finding_scores(f: dict) -> dict:
    """Backfill a finding's per_benchmark (CI/n) from the eval worker's authoritative
    scores.json — matched by idea_name + headline — so findings shared by agents that
    dropped fields still display full numbers (no '?'). Read-time only; never mutates
    the stored finding. No-op if already complete or no match."""
    cs = f.get("composite_scores") or {}
    pb = cs.get("per_benchmark") or {}
    if pb and all(isinstance(v, dict) and v.get("ci_low") is not None for v in pb.values()):
        return f  # already complete
    name = f.get("idea_name")
    sd = SCORES
    if not name or not sd.exists():
        return f
    want = f.get("headline_pct")
    if not isinstance(want, (int, float)):
        want = cs.get("headline_pct")
    cands = []
    for p in sd.glob(f"{name}*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(d, dict) and d.get("per_benchmark"):
            cands.append((p.stat().st_mtime, d))
    if not cands:
        return f
    cands.sort(key=lambda t: t[0])
    auth = None
    if isinstance(want, (int, float)):
        def _h(d):
            v = d.get("headline_pct")
            return v if isinstance(v, (int, float)) else -1e9
        best = min(cands, key=lambda t: abs(_h(t[1]) - want))
        if abs(_h(best[1]) - want) <= 0.2:
            auth = best[1]
    if auth is None:
        auth = cands[-1][1]
    apb = auth.get("per_benchmark") or {}
    merged_pb = dict(pb)
    for k, av in apb.items():
        cur = dict(merged_pb.get(k) or {})
        cur.update({kk: vv for kk, vv in av.items() if vv is not None})  # ci/n/mean from worker
        merged_pb[k] = cur
    cs = dict(cs)
    cs["per_benchmark"] = merged_pb
    for k in ("closed_pct", "filter_detail", "passes_filter", "headline_pct"):
        if cs.get(k) is None and auth.get(k) is not None:
            cs[k] = auth[k]
    f = dict(f)
    f["composite_scores"] = cs
    return f


def _finding(fid: str) -> dict:
    """Full finding for the detail drawer, plus its code snapshot's files (inline
    text for small source files). Matches by id or post_id."""
    fdir = _forum_dir()
    config.LOCAL_FINDINGS_DIR = str(fdir)
    match = None
    for f in fs_forum.read_findings():
        if fid and (f.get("id") == fid or f.get("post_id") == fid):
            match = f
            break
    if match is None:
        return {"error": "not found"}
    match = _enrich_finding_scores(match)
    baselines = _baselines()
    code = []
    snap = match.get("code_snapshot")
    if snap:
        d = fdir / snap
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if not p.is_file():
                    continue
                size = p.stat().st_size
                info = {"name": str(p.relative_to(d)), "size": size}
                if p.suffix in _CODE_TEXT_EXTS and size <= _CODE_MAX_INLINE_BYTES:
                    try:
                        info["content"] = p.read_text(errors="ignore")
                    except Exception:
                        pass
                code.append(info)
    return {"finding": match, "code": code, "baselines": baselines}


def _qualifies(f: dict) -> bool:
    """A 'clean win': improves EVERY safety benchmark over baseline (all closed% > 0)
    AND degrades NO capability benchmark (passes the 3-way capability filter).
    Stricter than the headline alone — this is what advances the best-so-far line."""
    cs = f.get("composite_scores") or {}
    closed = cs.get("closed_pct") or {}
    if not closed or not cs.get("passes_filter"):
        return False
    return all(isinstance(v, (int, float)) and v > 0 for v in closed.values())


def _headline(f: dict):
    cs = f.get("composite_scores") or {}
    h = f.get("headline_pct")
    if not isinstance(h, (int, float)):
        h = cs.get("headline_pct")
    return h if isinstance(h, (int, float)) else None


def _chain_of(f: dict) -> str:
    uid = f.get("idea_uid") or ""
    for c in ("alpha", "beta", "gamma"):
        if c in uid:
            return c
    return "team"


def _best_so_far() -> dict:
    """Cumulative best 'clean win' over the run (time order). The team line is the
    running best qualifying headline; per-chain lines the same within each chain;
    every finding is also returned for the attempt scatter (filtered vs qualifying)."""
    findings = sorted(fs_forum.read_findings(), key=lambda f: f.get("created_at", ""))
    team, chains, points = [], {}, []
    tbest, cbest, best_meta, nq = None, {}, None, 0
    for i, f in enumerate(findings, 1):
        h, q, c, nm = _headline(f), _qualifies(f), _chain_of(f), f.get("idea_name")
        if q:
            nq += 1
            if h is not None and (tbest is None or h > tbest):
                tbest = h
                best_meta = {"score": round(h, 2), "idea_name": nm, "chain": c, "i": i}
            if h is not None and (cbest.get(c) is None or h > cbest[c]):
                cbest[c] = h
        team.append({"x": i, "y": (round(tbest, 2) if tbest is not None else None)})
        chains.setdefault(c, []).append({"x": i, "y": (round(cbest[c], 2) if cbest.get(c) is not None else None)})
        points.append({"x": i, "score": (round(h, 2) if h is not None else None), "q": q, "chain": c, "name": nm})
    return {"n_total": len(findings), "n_qualifying": nq, "best": best_meta,
            "team": team, "chains": chains, "points": points}


_BSF_CACHE = {"key": None, "png": b""}


def _render_bsf_png() -> bytes:
    """Render the best-so-far figure with matplotlib THROUGH Bruce's actual
    style.py (real cmcrameri.batlow palette, Times serif, inch-anchored
    apply_layout, fixed-size better_arrow, figure-level framed legend). This is
    the guideline applied for real, not an SVG imitation. Cached on the data
    signature so we don't re-render on every poll."""
    config.LOCAL_FINDINGS_DIR = str(_forum_dir())   # point fs_forum at the current team (independent of /api/state call order)
    b = _best_so_far()
    key = (b["n_total"], b["n_qualifying"], (b["best"] or {}).get("i"))
    if _BSF_CACHE["key"] == key and _BSF_CACHE["png"]:
        return _BSF_CACHE["png"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from aar.web_ui import bruce_style as bs

    bs.setup_rcparams()
    pal = bs.palette()
    # Per-chain color + marker for ANY number of AARs (not just 3): sample the batlow
    # colormap at N evenly-spaced positions and cycle through distinct markers.
    # Show EVERY AAR in the team — chains with findings get real lines; chains that are
    # running but haven't produced a finding yet still appear (flat at 0), so a 5-AAR team
    # shows all 5. Union of finding-chains + live squeue chains.
    _jp = _job_prefix()
    _running = {jn.replace(_jp, "") for jn in _squeue_jobs().keys()}
    # squeue can transiently return nothing (busy controller / timeout); also derive
    # launched chains from their .out logs (always on disk) so every AAR shows even
    # when a squeue poll hiccups — was dropping OLMo to 3 AARs.
    for _p in LOGS.glob(f"{_jp}*.out"):
        if not _out_in_pinned_team(_p):
            continue
        _m = re.match(rf"{re.escape(_jp)}([^_]+)_\d+\.out", _p.name)
        if _m:
            _running.add(_m.group(1))
    # Drop a pseudo-chain literally named "team": the team aggregate already has its own
    # highlighted line + "team best" legend entry, so it must not also appear as an AAR.
    chain_names = sorted((set(b["chains"].keys()) | _running) - {"team"})
    _markers = ["o", "D", "s", "^", "v", "P", "X", "*", "p", "h"]
    _n = max(1, len(chain_names))
    try:
        _cmap = bs._BATLOW
        chain_col = {nm: _cmap(0.18 + 0.64 * (i / (_n - 1) if _n > 1 else 0.5))
                     for i, nm in enumerate(chain_names)}
    except Exception:
        _pk = ["primary", "secondary", "tertiary"]
        chain_col = {nm: pal[_pk[i % len(_pk)]] for i, nm in enumerate(chain_names)}
    chain_mk = {nm: _markers[i % len(_markers)] for i, nm in enumerate(chain_names)}
    FIG_W, FIG_H = bs.figsize_for(bs.FULL_PAGE_W, n_rows=1)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    aar_handles = []      # per-AAR colour/marker key (lower legend tier)
    summary_handles = []  # team-best + filtered (upper framed legend tier)
    if not b["n_total"]:
        ax.text(0.5, 0.5, "no methods evaluated yet", ha="center", va="center",
                transform=ax.transAxes, color=bs.NEUTRAL)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        pts = b["points"]
        # Drop OUTLIER points (>3 std from the mean of all method scores) so a single wild
        # run — e.g. a method that wrecked the model and scored a hugely negative headline —
        # doesn't blow out the y-axis and squash the meaningful range. Needs >=4 points.
        _sc = [p["score"] for p in pts if p["score"] is not None]
        if len(_sc) >= 4:
            _m = sum(_sc) / len(_sc)
            _sd = (sum((s - _m) ** 2 for s in _sc) / len(_sc)) ** 0.5
            _inlier = lambda s: _sd == 0 or abs(s - _m) <= 3 * _sd
        else:
            _inlier = lambda s: True
        fx = [p["x"] for p in pts if not p["q"] and p["score"] is not None and _inlier(p["score"])]
        fy = [p["score"] for p in pts if not p["q"] and p["score"] is not None and _inlier(p["score"])]
        if fx:
            ax.scatter(fx, fy, s=16, facecolors="none", edgecolors=bs.NEUTRAL,
                       linewidths=0.8, zorder=2)
        _max_x = max((p["x"] for p in pts), default=1)
        for c in chain_names:
            col, mk = chain_col[c], chain_mk[c]
            cp = b["chains"].get(c)
            # Anchor every line at (iteration 1, 0%): chains with findings step up from 0; a
            # running chain with no finding yet shows a flat line at 0 across the iterations.
            if cp:
                sx = [1] + [p["x"] for p in cp]
                sy = [0.0] + [(p["y"] if p["y"] is not None else 0.0) for p in cp]
            else:
                sx, sy = [1, _max_x], [0.0, 0.0]
            ax.step(sx, sy, where="post", color=col, lw=1.0, alpha=0.55, zorder=3)
            qx = [p["x"] for p in pts if p["q"] and p["chain"] == c and p["score"] is not None and _inlier(p["score"])]
            qy = [p["score"] for p in pts if p["q"] and p["chain"] == c and p["score"] is not None and _inlier(p["score"])]
            if qx:
                ax.plot(qx, qy, linestyle="none", marker=mk, ms=5, color=col,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=4)
            aar_handles.append(Line2D([0], [0], color=col, marker=mk, ms=5,
                                  markeredgecolor="white", lw=1.4, label=f"AAR {c}"))
        tx = [1] + [p["x"] for p in b["team"]]
        ty = [0.0] + [(p["y"] if p["y"] is not None else 0.0) for p in b["team"]]
        if len(tx) > 1:
            # team best = THE headline line: thick + opaque + on top, with a soft white
            # halo so it reads clearly over the muted per-AAR lines.
            import matplotlib.patheffects as _pe
            ax.step(tx, ty, where="post", color=pal["highlight"], lw=3.3, zorder=6,
                    solid_capstyle="round", solid_joinstyle="round",
                    path_effects=[_pe.Stroke(linewidth=5.2, foreground="white"), _pe.Normal()])
        summary_handles.append(Line2D([0], [0], color=pal["highlight"], lw=3.3, label="team best (all AARs)"))
        summary_handles.append(Line2D([0], [0], color=bs.NEUTRAL, marker="o", ms=5,
                              markerfacecolor="none", linestyle="none", label="filtered (excluded)"))
        ax.axhline(0, color=bs.NEUTRAL, lw=0.8, linestyle=(0, (3, 2)), zorder=1)
        ax.set_xlabel("iterations")
        ax.set_ylabel("Safety Geometric Mean (%)")
        ax.margins(x=0.03)
        # Fixed y-axis floor at 0%: under the geometric-mean headline every score is
        # clamped to [0,1], so the headline is never negative — 0 is the natural floor
        # (a method that left some attack at/below baseline scores 0). Top stays data-driven.
        ax.autoscale_view()
        ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 5))
    # Two-tier legend (keeps the figure narrow even with many AARs): a framed SUMMARY
    # row (team best + filtered — what the lines MEAN) on top, and a frameless per-AAR
    # colour KEY below it (which colour = which AAR). apply_layout sets the base single-
    # panel layout; we then push the plot down to make room for the stacked legend.
    bs.apply_layout(fig, FIG_H, n_rows=1, n_cols=1)
    if summary_handles or aar_handles:
        key_ncol = min(len(aar_handles), 6) or 1
        key_rows = -(-len(aar_handles) // key_ncol)        # ceil division (0 if no AARs)
        # reserve top room: framed summary row (~0.40") + gap + ~0.20" per per-AAR key row
        top_in = 0.40 + (0.18 + 0.20 * key_rows if aar_handles else 0.0)
        fig.subplots_adjust(top=1 - top_in / FIG_H, bottom=bs.BOTTOM_PAD / FIG_H)
        if summary_handles:
            _leg = fig.legend(handles=summary_handles, loc="upper center",
                              bbox_to_anchor=(0.5, 1 - 0.14 / FIG_H), ncol=len(summary_handles),
                              frameon=True, fancybox=True, framealpha=0.95, handletextpad=0.4,
                              columnspacing=1.6, borderpad=0.4)
            fig.add_artist(_leg)   # keep it when the second fig.legend is added
        if aar_handles:
            fig.legend(handles=aar_handles, loc="upper center",
                       bbox_to_anchor=(0.5, 1 - 0.40 / FIG_H), ncol=key_ncol, frameon=False,
                       handletextpad=0.35, columnspacing=1.1, title="per-AAR best",
                       title_fontsize=7)
    if b["n_total"]:
        try:
            bs.better_arrow(ax, direction="up", corner="lower right")
        except Exception:
            pass
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    png = buf.getvalue()
    _BSF_CACHE.update(key=key, png=png)
    return png


def _state() -> dict:
    fdir = _forum_dir()
    config.LOCAL_FINDINGS_DIR = str(fdir)   # point fs_forum at the current team's forum
    findings = sorted(fs_forum.read_findings(), key=lambda f: f.get("created_at", ""), reverse=True)
    return {
        "team": fdir.name,
        "chains": _chain_status(),
        "leaderboard": fs_forum.leaderboard(limit=50),
        "findings": findings[:60],
        "findings_total": len(findings),
        "literature": _lit_entries(),
        "lit_agent": _lit_agent(),
        "scores": _scores(),
        "baselines": _baselines(),
        "best_so_far": _best_so_far(),
        "now": time.strftime("%H:%M:%S"),
    }


# Axis label shown in the dashboard chrome — driven by the live suite (no hardcoded axis),
# so the same dashboard serves any safety axis the harness is run for.
def _render_page() -> str:
    """The page with the axis + model substituted (title + header). Resolved at call time
    (not import) so it reflects the live team regardless of import order. Title is
    'AAR: <axis> X <model> Team' — axis/model parsed from the team_id (<axis>-<model>-<ts>),
    with axis falling back to SUITE_NAME and model to '?' if the id carries no model."""
    axis = os.getenv("SUITE_NAME") or getattr(config, "SUITE_NAME", "safety")
    model = "?"
    base = re.sub(r"-\d{8}-\d{6}(-\d+)?$", "", _team_id())   # strip timestamp[-n] -> <axis>-<model>[-<agent>]
    if "-" in base:
        parts = base.split("-")
        axis, model = parts[0], parts[1]
        if len(parts) >= 3:                                  # agent-model tag present -> surface it
            model = f"{parts[1]} ({parts[2]})"
    return PAGE.replace("__AXIS__", axis).replace("__MODEL__", model)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>AAR: __AXIS__ X __MODEL__ Team</title>
<style>
 /* Bruce light/white aesthetic: white bg, serif type, batlow-role accents, light grid. */
 /* warm parchment page, white cards/figures that lift off it — pairs with batlow + serif */
 :root{--bg:#f3f1ea;--card:#ffffff;--bd:#e1dccf;--fg:#22201b;--mut:#726c60;--ok:#2ca02c;--bad:#d62728;--acc:#5a6b2f;--code:#f6f3ec;--tint:rgba(108,124,60,.10)}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 'Times New Roman','Times','DejaVu Serif',Georgia,serif}
 header{display:flex;align-items:baseline;gap:14px;padding:14px 20px;border-bottom:1px solid var(--bd);position:sticky;top:0;background:var(--bg);z-index:5}
 h1{font-size:19px;margin:0} .mut{color:var(--mut)} .wrap{padding:16px 20px;display:grid;grid-template-columns:1.1fr 1fr;gap:16px}
 .full{grid-column:1/-1} .card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:14px;box-shadow:0 1px 2px rgba(27,31,36,.04)}
 h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 10px;font-family:Georgia,serif}
 .chain{display:flex;justify-content:space-between;gap:10px;padding:8px 6px;border-bottom:1px solid var(--bd);cursor:pointer;border-radius:6px}
 .chain:last-child{border:0} .chain:hover{background:var(--tint)} .name{font-weight:700;color:var(--acc)}
 .bar{height:6px;background:#eef1f4;border-radius:4px;margin-top:4px;overflow:hidden}
 .bar>i{display:block;height:100%;background:var(--acc)} .st{font-size:12px;padding:2px 8px;border-radius:20px;border:1px solid var(--bd)}
 .RUNNING{color:var(--ok);border-color:var(--ok)} .done,.GONE{color:var(--mut)}
 table{width:100%;border-collapse:collapse} td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--bd);font-size:13px}
 th{color:var(--mut);font-weight:600} .pos{color:var(--ok)} .neg{color:var(--bad)} .pill{font-size:11px;padding:1px 7px;border-radius:20px}
 .pass{background:rgba(44,160,44,.13);color:#1f7a1f} .fail{background:rgba(214,39,40,.12);color:var(--bad)}
 .find{padding:10px 6px;border-bottom:1px solid var(--bd);cursor:pointer;border-radius:6px} .find:last-child{border:0} .find:hover{background:var(--tint)}
 .find .h{display:flex;gap:8px;align-items:baseline} .ty{font-size:11px;color:var(--mut)} .sum{color:var(--mut);font-size:12px;margin-top:3px;max-height:3.6em;overflow:hidden}
 .last{color:var(--mut);font-size:12px;font-family:ui-monospace,monospace;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
 .empty{color:var(--mut);padding:8px 0}
 .view{color:var(--acc);font-size:12px}
 .countpill{display:inline-block;background:var(--acc);color:#fff;font-size:10.5px;font-weight:700;padding:1.5px 9px;border-radius:20px;margin-left:8px;vertical-align:middle;letter-spacing:.02em;text-transform:none;box-shadow:0 1px 2px rgba(0,0,0,.12)}
 tr.lbrow{cursor:pointer} tr.lbrow:hover td{background:var(--tint)}
 #bsfimg{width:100%;max-width:660px;display:block;margin:6px auto;border:1px solid var(--bd);border-radius:6px;background:#fff}
 #ov{position:fixed;inset:0;background:rgba(27,31,36,.35);display:none;z-index:9}
 #dw{position:fixed;top:0;right:0;width:min(860px,94vw);height:100%;background:var(--card);border-left:1px solid var(--bd);overflow:auto;padding:18px 20px;box-shadow:-10px 0 40px rgba(27,31,36,.18)}
 #dw h3{margin:6px 0 10px} .x{float:right;cursor:pointer;color:var(--mut);font-size:22px;line-height:1}
 .sess{padding:9px 11px;border:1px solid var(--bd);border-radius:8px;margin:6px 0;cursor:pointer;display:flex;justify-content:space-between}
 .sess:hover{border-color:var(--acc)} .back{color:var(--acc);cursor:pointer;font-size:13px}
 .ev{margin:10px 0;padding-left:11px;border-left:2px solid var(--bd)} .ev .et{font-size:11px;color:var(--mut)}
 .ev.think{border-color:var(--acc)} .ev.tool{border-color:#b5651d} .ev.res{border-color:var(--ok)}
 .ev .bd{white-space:pre-wrap;font-size:13.5px;margin-top:3px}
 .ev.tool .bd,.ev.res .bd{font-family:ui-monospace,monospace;font-size:12px;color:#3a3f45;max-height:15em;overflow:auto;background:var(--code);border:1px solid var(--bd);border-radius:6px;padding:8px}
 #dw h4{margin:15px 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
 #dw pre.md{white-space:pre-wrap;font:13px/1.55 'Times New Roman',Georgia,serif;background:var(--code);border:1px solid var(--bd);border-radius:6px;padding:11px;margin:0}
 #dw pre.code{white-space:pre;overflow:auto;font:12px ui-monospace,monospace;background:var(--code);border:1px solid var(--bd);border-radius:6px;padding:10px;max-height:32em;margin:6px 0 0}
 #dw details{margin:5px 0;border:1px solid var(--bd);border-radius:6px;padding:7px 11px;background:var(--code)}
 #dw summary{cursor:pointer;font-family:ui-monospace,monospace;font-size:12.5px} #dw table{margin:2px 0}
 #litpane{max-height:560px;overflow:auto;column-count:2;column-gap:18px}
 @media(max-width:900px){#litpane{column-count:1}}
 .lit{break-inside:avoid;padding:9px 4px;border-bottom:1px solid var(--bd)} .lit:last-child{border:0}
 .lit .h{display:flex;gap:7px;align-items:baseline;flex-wrap:wrap} .lit .sum{color:var(--mut);font-size:12.5px;margin-top:3px}
 .relp{font-size:10px;padding:1px 7px;border-radius:20px;text-transform:uppercase;letter-spacing:.04em}
 .rel-high{background:rgba(44,160,44,.14);color:#1f7a1f} .rel-medium{background:var(--tint);color:var(--acc)} .rel-low{background:#ececec;color:var(--mut)}
 .lit details{margin-top:4px} .lit summary{cursor:pointer;color:var(--acc);font-size:12px}
 .litd{font-size:12px;color:#3a3f45;margin:5px 0;padding-left:4px}
 .litd.litr{background:var(--tint);border-radius:4px;padding:5px 7px}
 .littail{font-family:ui-monospace,monospace;font-size:11px;color:var(--mut);margin-top:4px;white-space:pre-wrap;border-left:2px solid var(--bd);padding-left:8px;max-height:7em;overflow:auto}
 .lit{cursor:pointer;border-radius:6px} .lit:hover{background:var(--tint)}
 #pop{position:fixed;inset:0;background:rgba(27,31,36,.4);display:none;z-index:20;align-items:center;justify-content:center}
 #popbox{background:var(--card);border:1px solid var(--bd);border-radius:10px;width:min(660px,92vw);max-height:82vh;overflow:auto;padding:16px 22px;box-shadow:0 16px 50px rgba(27,31,36,.32);position:relative}
 #popbox h3{margin:2px 28px 10px 0;font-size:16px} #popbox .x{position:absolute;top:12px;right:16px;cursor:pointer;color:var(--mut);font-size:24px;line-height:1}
 #popbox .litd{font-size:13px;color:#2c2f34;margin:9px 0;line-height:1.55} #popbox .litd b{color:var(--acc)}
 #popbox .litd.litr{background:var(--tint);border-radius:6px;padding:8px 10px}
 #poplog{white-space:pre-wrap;font:11.5px/1.5 ui-monospace,monospace;color:#3a3f45;background:var(--code);border:1px solid var(--bd);border-radius:6px;padding:10px;max-height:64vh;overflow:auto;margin-top:6px}
</style></head><body>
<header><h1>AAR: __AXIS__ X __MODEL__ Team</h1>
 <span class="mut" id="meta">connecting…</span></header>
<div class="wrap">
 <div class="card full"><h2>Main Result</h2><div id="bsfcap" class="mut" style="font-size:12px;margin-bottom:2px"></div><img id="bsfimg" alt="best-so-far figure (rendered via Bruce's matplotlib style)"></div>
 <div class="card full"><h2>AARs</h2><div id="chains"></div></div>
 <div class="card"><h2>Leaderboard <span class="mut">(Safety Geometric Mean = geometric mean of headroom-closed over the 3 safety benchmarks)</span></h2><div id="lb"></div></div>
 <div class="card"><h2>Forum — latest findings <span id="feedcount" class="countpill"></span></h2><div id="feed"></div></div>
 <div class="card full"><h2>Literature forum <span class="mut" id="litmeta"></span></h2>
  <div id="litagent" class="mut" style="font-size:12px;margin-bottom:8px">librarian: —</div>
  <div id="litpane"></div></div>
</div>
<div id="ov" onclick="if(event.target.id=='ov')closeOv()"><div id="dw"></div></div>
<div id="pop" onclick="if(event.target.id=='pop')closePop()"><div id="popbox"></div></div>
<script>
const $=s=>document.querySelector(s);
const pct=v=>v==null?'—':(v>=0?'+':'')+(+v).toFixed(2)+'%';
const cls=v=>v==null?'':(v>=0?'pos':'neg');
let BL={};  // per-benchmark baselines {name:{baseline,floor}} — base-model scores for side-by-side display
function chains(cs){ if(!cs.length) return '<div class="empty">no chains found</div>';
 return cs.map(c=>{const f=Math.min(100,(c.iterations/c.max_iters*100)||0);
  return `<div class="chain" onclick="openChain('${c.name}')"><div style="flex:1"><span class="name">${c.name}</span>
   <span class="mut"> · <b style="color:var(--fg)">${c.methods}</b> methods · ${c.shares} shared · session ${c.iterations}/${c.max_iters} · ${c.elapsed||''}</span> <span class="view">view trajectory ›</span>
   <div class="bar"><i style="width:${f}%"></i></div>
   <div class="last">${(c.last||'').replace(/</g,'&lt;')}</div></div>
   <span class="st ${c.state}">${c.state}</span></div>`}).join('')}
const shortname=k=>k;  /* show full benchmark names — axis-agnostic (no hardcoded per-axis map) */
const f3=v=>v==null?'—':(+v).toFixed(3);
function lb(rows){ if(!rows.length) return '<div class="empty">no results yet — chains are training…</div>';
 const head='<tr><th>#</th><th>method</th><th>Safety Geometric Mean</th><th>filter</th><th>raw per-benchmark score (closed%)</th></tr>';
 return '<table>'+head+rows.map((r,i)=>{const cs=r.composite_scores||{};const cl=cs.closed_pct||{};const pb=cs.per_benchmark||{};
  const pf=cs.passes_filter;
  // RAW score for every evaluated benchmark, with the closed% in parens for safety ones.
  const per=Object.entries(pb).map(([k,v])=>{const c=cl[k];
    const b=(v&&v.baseline!=null)?v.baseline:((BL[k]&&BL[k].baseline!=null)?BL[k].baseline:null);
    return `${shortname(k)} <b>${f3(v&&v.mean)}</b>${b!=null?`<span class="mut">/${f3(b)}</span>`:''}${c!=null?` <span class="${cls(c)}">(${pct(c)})</span>`:''}`;}).join(' · ');
  return `<tr class="lbrow" title="click for full idea details" onclick="openFinding('${r.id||r.post_id||''}')"><td>${i+1}</td><td><b>${r.idea_name||'—'}</b> <span class="view">›</span></td>
   <td class="${cls(r.headline_pct)}"><b>${pct(r.headline_pct)}</b></td>
   <td>${pf==null?'':`<span class="pill ${pf?'pass':'fail'}">${pf?'PASS':'FAIL'}</span>`}</td>
   <td class="mut">${per||'—'}</td></tr>`}).join('')+'</table>'}
function feed(fs){ if(!fs.length) return '<div class="empty">no findings yet</div>';
 return fs.map(f=>`<div class="find" onclick="openFinding('${f.id||''}')"><div class="h"><span class="name">${f.idea_name||'—'}</span>
  <span class="${cls(f.headline_pct)}">${f.headline_pct!=null?pct(f.headline_pct):''}</span>
  <span class="ty">· ${f.finding_type||''} · ${(f.created_at||'').replace('T',' ').replace('Z','')}</span>
  <span class="view" style="margin-left:auto">open ›</span></div>
  ${(()=>{const pb=(f.composite_scores||{}).per_benchmark||{};const e=Object.entries(pb);
    return e.length?`<div class="last">${e.map(([k,v])=>{const b=(v&&v.baseline!=null)?v.baseline:((BL[k]&&BL[k].baseline!=null)?BL[k].baseline:null);
      return `${shortname(k)} ${f3(v&&v.mean)}${b!=null?'/'+f3(b):''}`;}).join(' · ')} <span class="mut">(score/base)</span></div>`:'';})()}
  <div class="sum">${((f.title?f.title+' — ':'')+(f.summary||'')).replace(/[#*`]/g,'').replace(/</g,'&lt;').slice(0,260)}</div></div>`).join('')}
const RELO={high:2,medium:1,low:0};
let LITS=[], LITA={}, popMode=null;
function closePop(){ $('#pop').style.display='none'; popMode=null; }
document.addEventListener('keydown',e=>{if(e.key=='Escape')closePop()});
function litforum(es){ if(!es.length) return '<div class="empty">no literature yet — the librarian is surveying the field…</div>';
 const order=es.map((e,i)=>[e,i]).sort((a,b)=>(RELO[(b[0].relevance||'').toLowerCase()]||0)-(RELO[(a[0].relevance||'').toLowerCase()]||0));
 return order.map(([e,i])=>{const rel=(e.relevance||'').toLowerCase();
  return `<div class="lit" onclick="openLitEntry(${i})"><div class="h"><span class="name">${esc(e.method||'—')}</span>
   ${e.relevance?`<span class="relp rel-${rel}">${esc(e.relevance)}</span>`:''}
   <span class="ty">· ${esc(e.category||'')}</span><span class="view" style="margin-left:auto">details ›</span></div>
   <div class="sum">${esc(e.summary||'')}</div></div>`;}).join('')}
function openLitEntry(i){ const e=LITS[i]; if(!e) return; popMode='entry';
 const row=(lbl,v,hl)=>v?`<div class="litd${hl?' litr':''}"><b>${lbl}.</b> ${esc(v)}</div>`:'';
 $('#popbox').innerHTML=`<span class="x" onclick="closePop()">×</span>
  <h3>${esc(e.method||'—')} ${e.relevance?`<span class="relp rel-${(e.relevance||'').toLowerCase()}">${esc(e.relevance)}</span>`:''} <span class="ty mut">· ${esc(e.category||'')}</span></h3>
  ${row('Summary',e.summary)}${row('Intuition',e.intuition)}${row('Mechanism',e.core_mechanism)}
  ${row('Reproduction recipe',e.reproduction_recipe,1)}${row('Prerequisites',e.prerequisites)}
  ${row('Training data',e.training_data)}${row('Evaluation',e.evaluation)}${row('Results',e.key_results,1)}
  ${row('Applicability',e.applicability)}${row('Source',e.source)}`;
 $('#pop').style.display='flex'; $('#popbox').scrollTop=0;}
function openLitAgent(){ popMode='agent'; renderLitAgentPop(); $('#pop').style.display='flex'; }
function renderLitAgentPop(){ const la=LITA||{}; const log=(la.log&&la.log.length)?la.log:(la.progress||[]);
 $('#popbox').innerHTML=`<span class="x" onclick="closePop()">×</span>
  <h3>Librarian <span class="st ${la.state||'—'}">${la.state||'—'}</span> <span class="mut" style="font-size:12px">${la.elapsed?'· '+esc(la.elapsed):''}</span></h3>
  <div class="mut" style="font-size:11px;margin-bottom:2px">literature-review trajectory — live</div>
  <div id="poplog">${log.map(l=>esc(l)).join('\n')}</div>`;
 const pl=$('#poplog'); if(pl) pl.scrollTop=pl.scrollHeight;}
function litagentHtml(la){ LITA=la||{}; const st=la.state||'—';
 const last=(la.progress||[]).slice(-2).map(l=>esc(l)).join('\n');
 return `<span class="st ${st}" style="cursor:pointer" onclick="openLitAgent()" title="click for the librarian's live trajectory">librarian: ${st}</span>`+
  `${la.elapsed?' · '+esc(la.elapsed):''} <span class="view" style="font-size:11px;cursor:pointer" onclick="openLitAgent()">view trajectory ›</span>${last?`<div class="littail">${last}</div>`:''}`;}
// Best-so-far figure is rendered server-side by matplotlib via Bruce's real
// style.py (see _render_bsf_png) and served at /api/bsf.png — refreshed below.
async function tick(){ try{const r=await fetch('/api/state');const d=await r.json();
  BL=d.baselines||{};
  $('#meta').textContent='team '+(d.team||'—')+' · '+d.chains.length+' AARs · '+d.leaderboard.length+' results · updated '+d.now;
  const bb=d.best_so_far||{};
  $('#bsfcap').innerHTML = bb.best?`best clean win: <b style="color:var(--fg)">${bb.best.score}%</b> — ${esc(bb.best.idea_name)} (${bb.best.chain}) · ${bb.n_qualifying}/${bb.n_total} qualify`:`${bb.n_qualifying||0}/${bb.n_total||0} qualify`;
  $('#bsfimg').src='/api/bsf.png?t='+Date.now();
  $('#chains').innerHTML=chains(d.chains); $('#lb').innerHTML=lb(d.leaderboard); $('#feed').innerHTML=feed(d.findings);
  $('#feedcount').textContent=(d.findings_total||0);
  const lit=d.literature||[]; LITS=lit; LITA=d.lit_agent||{};
  $('#litpane').innerHTML=litforum(lit);
  $('#litmeta').textContent='· '+lit.length+' entries'; $('#litagent').innerHTML=litagentHtml(LITA);
  if(popMode=='agent') renderLitAgentPop();
 }catch(e){ $('#meta').textContent='disconnected — retrying'; } }
const ov=$('#ov'),dw=$('#dw'),esc=s=>(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');
function closeOv(){ov.style.display='none'}
document.addEventListener('keydown',e=>{if(e.key=='Escape')closeOv()});
async function openChain(name){ ov.style.display='block';
 dw.innerHTML=`<span class="x" onclick="closeOv()">×</span><h3>${name} — iterations</h3><div class="mut">loading…</div>`;
 try{const d=await(await fetch('/api/sessions?chain='+encodeURIComponent(name))).json(); const ss=d.sessions||[];
  dw.innerHTML=`<span class="x" onclick="closeOv()">×</span><h3>${name} — iterations</h3>`+
   (ss.length?ss.map(s=>`<div class="sess" onclick="openSession('${s.id}','${name}',${s.index})">
     <span>iteration ${s.index}</span><span class="mut">${s.n_events} events${s.exists?'':' · no transcript'}</span></div>`).join('')
    :'<div class="empty">no iterations recorded yet</div>');
 }catch(e){dw.innerHTML+='<div class="empty">error loading</div>'} }
async function openSession(id,name,idx){
 const hdr=`<span class="x" onclick="closeOv()">×</span><span class="back" onclick="openChain('${name}')">‹ ${name}</span><h3>iteration ${idx} <span class="mut">— full trajectory</span></h3>`;
 dw.innerHTML=hdr+'<div class="mut">loading transcript…</div>'; dw.scrollTop=0;
 try{const d=await(await fetch('/api/session?id='+encodeURIComponent(id))).json();
  dw.innerHTML=hdr+renderEvents(d.events||[]);
 }catch(e){dw.innerHTML=hdr+'<div class="empty">error loading transcript</div>'} }
function renderEvents(evs){ if(!evs.length) return '<div class="empty">transcript not captured for this iteration</div>';
 return evs.map(e=>{let cls='',lbl=e.type,bd=e.body||'';
  if(e.type=='AssistantMessage'&&bd.startsWith('Tool:')){cls='tool';lbl='→ '+(bd.split('\n')[0].replace('Tool: ',''));}
  else if(e.type=='AssistantMessage'){cls='think';lbl='💭 reasoning';}
  else if(e.type=='UserMessage'){cls='res';lbl='result / observation';}
  else{return '';}
  if(!bd) return '';
  return `<div class="ev ${cls}"><div class="et">${e.ts} · ${lbl}</div><div class="bd">${esc(bd)}</div></div>`;}).join('')}
async function openFinding(id){ if(!id) return; ov.style.display='block';
 dw.innerHTML=`<span class="x" onclick="closeOv()">×</span><div class="mut">loading finding…</div>`; dw.scrollTop=0;
 try{const d=await(await fetch('/api/finding?id='+encodeURIComponent(id))).json();
  if(d&&d.baselines)BL=d.baselines;
  dw.innerHTML=d&&d.finding?renderFinding(d.finding,d.code||[]):`<span class="x" onclick="closeOv()">×</span><div class="empty">finding not found</div>`;
 }catch(e){dw.innerHTML=`<span class="x" onclick="closeOv()">×</span><div class="empty">error loading finding</div>`} }
function renderFinding(f,code){
 const cs=f.composite_scores||{}, cl=cs.closed_pct||{}, fil=cs.filter_detail||{}, pb=cs.per_benchmark||{};
 // RAW score for EVERY evaluated benchmark (mean + 95% CI + n), safety and capability alike.
 const raw=Object.entries(pb).map(([k,v])=>{const bl=BL[k]||{};
   const base=(v.baseline!=null)?v.baseline:(bl.baseline!=null?bl.baseline:null);
   const floor=(v.floor!=null)?v.floor:(bl.floor!=null?bl.floor:(bl.baseline_ci_low!=null?bl.baseline_ci_low:null));
   const baseCell=base!=null?(+base).toFixed(3)+(floor!=null?` <span class="mut">(floor ${(+floor).toFixed(3)})</span>`:''):'—';
   const ciCell=(v.ci_low!=null&&v.ci_high!=null)?`[${(+v.ci_low).toFixed(3)}, ${(+v.ci_high).toFixed(3)}]`:'—';
   const nCell=(v.n!=null)?('n='+v.n):'—';
   return `<tr><td>${k}</td><td><b>${v.mean!=null?(+v.mean).toFixed(4):'—'}</b></td>
   <td class="mut">${baseCell}</td>
   <td class="mut">${ciCell}</td>
   <td class="mut">${nCell}</td>${cl[k]!=null?`<td class="${cls(cl[k])}">${pct(cl[k])}</td>`:'<td class="mut">filter</td>'}</tr>`;}).join('');
 const per=Object.entries(cl).map(([k,v])=>`<tr><td>${k}</td><td class="${cls(v)}">${pct(v)}</td></tr>`).join('');
 const fmt=v=>v==null?'—':(+v).toFixed(3);
 const filt=Object.entries(fil).map(([k,v])=>`<tr><td>${k}</td><td>${fmt(v.mean)}</td>
   <td class="mut">floor ${fmt(v.floor!=null?v.floor:v.baseline_ci_low)}</td>
   <td><span class="pill ${v.passed?'pass':'fail'}">${v.passed?'PASS':'FAIL'}</span></td></tr>`).join('');
 const cfg=f.config?`<h4>config</h4><pre class="code">${esc(JSON.stringify(f.config,null,2))}</pre>`:'';
 const codeHtml=code.length?'<h4>method code <span class="mut">('+code.length+' file'+(code.length>1?'s':'')+', from snapshot)</span></h4>'+
   code.map(c=>`<details><summary>${esc(c.name)} <span class="mut">· ${c.size}b</span></summary>${
     c.content!=null?`<pre class="code">${esc(c.content)}</pre>`:'<div class="mut" style="margin-top:6px">[binary or &gt;60kb — not inlined]</div>'}</details>`).join(''):'';
 return `<span class="x" onclick="closeOv()">×</span>
  <h3 style="margin-right:24px">${esc(f.idea_name||'—')}
   <span class="${cls(f.headline_pct)}">${f.headline_pct!=null?pct(f.headline_pct):''}</span>
   ${cs.passes_filter!=null?`<span class="pill ${cs.passes_filter?'pass':'fail'}">${cs.passes_filter?'PASS filter':'FAIL filter'}</span>`:''}</h3>
  <div class="mut">${esc(f.title||'')}</div>
  <div class="mut" style="margin-top:4px;font-size:12px">${f.finding_type||''} · ${(f.created_at||'').replace('T',' ').replace('Z','')} · id ${esc((f.id||'').slice(0,12))}</div>
  ${raw?`<h4>raw per-benchmark scores <span class="mut">(method vs base model)</span></h4><table><tr><th>benchmark</th><th>score</th><th>baseline</th><th>95% CI</th><th>n</th><th>closed</th></tr>${raw}</table>`:''}
  ${per?`<h4>safety — closed %</h4><table>${per}</table>`:''}
  ${filt?`<h4>capability filter</h4><table>${filt}</table>`:''}
  ${[['abstract','Abstract'],['motivation','Motivation'],['related_work','Related work'],['method','Method (objective, loss, math)'],['data','Data'],['experimental_setup','Experimental setup'],['results_writeup','Results write-up'],['hypothesis','Hypothesis & mechanism'],['methodology','Method (objective, loss, math)'],['data_recipe','Data recipe'],['generalization','Generalization'],['diagnosis','Diagnosis'],['next_steps','Next steps']]
     .filter(([k])=>f[k]).map(([k,lbl])=>`<h4>${lbl}</h4><pre class="md">${esc(f[k])}</pre>`).join('')}
  ${f.summary?`<h4>summary / notes</h4><pre class="md">${esc(f.summary)}</pre>`:''}
  ${cfg}${codeHtml}`;}
tick(); setInterval(tick,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/api/state":
                self._send(json.dumps(_state()).encode(), "application/json")
            elif u.path == "/api/sessions":
                self._send(json.dumps({"sessions": _chain_sessions(q.get("chain", [""])[0])}).encode(),
                           "application/json")
            elif u.path == "/api/session":
                sid = q.get("id", [""])[0]
                self._send(json.dumps({"id": sid, "events": _session_events(sid)}).encode(),
                           "application/json")
            elif u.path == "/api/finding":
                self._send(json.dumps(_finding(q.get("id", [""])[0])).encode(),
                           "application/json")
            elif u.path == "/api/bsf.png":
                try:
                    png = _render_bsf_png()
                except Exception as e:
                    print(f"[dashboard] bsf render failed: {e}", flush=True)
                    png = b""
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
            else:
                self._send(_render_page().encode(), "text/html; charset=utf-8")
        except BrokenPipeError:
            pass

    def log_message(self, *a):  # quiet
        pass


# Fetch shim injected into the static export so the SAME UI works against flat
# files (no server): /api/* fetches are rewritten to the exported .json files.
_STATIC_SHIM = (
    "<script>(function(){var _f=window.fetch.bind(window);window.fetch=function(u,o){"
    "try{if(typeof u==='string'){"
    "if(u.indexOf('/api/state')===0)u='state.json';"
    "else if(u.indexOf('/api/finding?id=')===0)u='finding_'+decodeURIComponent(u.split('id=')[1].split('&')[0])+'.json';"
    "else if(u.indexOf('/api/sessions?chain=')===0)u='sessions_'+decodeURIComponent(u.split('chain=')[1].split('&')[0])+'.json';"
    "else if(u.indexOf('/api/session?id=')===0)u='session_'+decodeURIComponent(u.split('id=')[1].split('&')[0])+'.json';"
    "}}catch(e){}return _f(u,o);};})();</script>\n"
)


def export_static(out_dir: str) -> None:
    """Export a SELF-CONTAINED static snapshot of the dashboard to out_dir: the same
    HTML/JS, plus the figure PNG and every /api/* payload as a flat .json file. A
    fetch-shim rewrites the UI's /api calls to those files, so it stays fully
    interactive on a plain static host (e.g. SiteGround) with no server. Re-run
    periodically (cron) to refresh — that's the 'live update'."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 1) HTML: inject the fetch-shim after <body>; point the figure <img> at bsf.png.
    html = _render_page().replace("</head><body>", "</head><body>\n" + _STATIC_SHIM, 1)
    html = html.replace("/api/bsf.png", "bsf.png")   # <img> src (set directly, not via fetch)
    (out / "index.html").write_text(html)
    # 2) state + figure
    state = _state()
    (out / "state.json").write_text(json.dumps(state))
    try:
        (out / "bsf.png").write_bytes(_render_bsf_png())
    except Exception as e:
        print(f"[static-export] bsf render failed: {e}")
    # 3) every finding's full detail (incl. code snapshot), keyed by id AND post_id
    seen = set()
    for f in (state.get("findings") or []) + (state.get("leaderboard") or []):
        for key in (f.get("id"), f.get("post_id")):
            if not key or key in seen:
                continue
            seen.add(key)
            try:
                (out / f"finding_{key}.json").write_text(json.dumps(_finding(key)))
            except Exception as e:
                print(f"[static-export] finding {key} failed: {e}")
    # 4) per-AAR sessions + every session's full trajectory (the drill-down)
    for c in state.get("chains") or []:
        name = c.get("name")
        if not name:
            continue
        sess = _chain_sessions(name)
        (out / f"sessions_{name}.json").write_text(json.dumps({"sessions": sess}))
        for s in sess:
            sid = s.get("id")
            if sid:
                (out / f"session_{sid}.json").write_text(
                    json.dumps({"id": sid, "events": _session_events(sid)}))
    n = len(list(out.glob("*")))
    print(f"[static-export] wrote {n} files to {out} (team={state.get('team')}, "
          f"{len(state.get('findings') or [])} findings)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--static-export", metavar="DIR",
                    help="render a self-contained static snapshot to DIR and exit "
                         "(for publishing to a static host like SiteGround)")
    args = ap.parse_args()
    if args.static_export:
        export_static(args.static_export)
        return
    print(f"[dashboard] http://{args.host}:{args.port}  (forum at {RUNS}/findings)", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
