"""Filesystem-backed forum for parallel AAR chains on a shared cluster FS.

On a multi-node Slurm cluster there is no single localhost server that all
chains can reach, and a shared-FS SQLite DB with many concurrent writers is a
locking footgun. So when ``FORUM_BACKEND=fs`` (the default whenever ``LOCAL_MODE``
is set), the forum *is* the shared findings directory (``config.LOCAL_FINDINGS_DIR``
on VAST): ``share_finding`` writes one atomic JSON file per finding and
``get_leaderboard`` globs + ranks them. Every chain mounts the same directory,
so all N chains see each other's findings with no server, no network dependency,
and no single point of failure. This is the same shared-FS coordination model
the rest of the harness uses for submissions/scores/holdout.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from aar import config
from .findings_sync import save_finding_to_dir, finding_filename


# Method-code files we never want in a snapshot: caches, VCS, and (defensively)
# trained-model artifacts in case an agent saved weights inside its idea dir.
_SNAPSHOT_SKIP_DIRS = {"__pycache__", ".git", "outputs", "model", "models",
                       "checkpoints", "wandb", ".ipynb_checkpoints"}
_SNAPSHOT_SKIP_EXTS = (".pyc", ".pt", ".bin", ".safetensors", ".ckpt",
                       ".gguf", ".onnx", ".h5", ".pth")
_SNAPSHOT_MAX_FILE_BYTES = 5 * 1024 * 1024  # skip any single file > 5 MB


def _snapshot_ignore(dirpath: str, names: list[str]) -> set[str]:
    """copytree ignore: drop caches/VCS, model weights, and oversized blobs so a
    code snapshot stays small (KBs of source, not GBs of weights)."""
    skip: set[str] = set()
    for n in names:
        if n in _SNAPSHOT_SKIP_DIRS or n.endswith(_SNAPSHOT_SKIP_EXTS):
            skip.add(n)
            continue
        try:
            p = Path(dirpath) / n
            if p.is_file() and p.stat().st_size > _SNAPSHOT_MAX_FILE_BYTES:
                skip.add(n)
        except OSError:
            pass
    return skip


def _snapshot_method_code(idea_name: str, dest_dir: Path) -> str | None:
    """Copy the method's source package ``aar/ideas/<idea_name>/`` into the forum
    at ``dest_dir`` so the finding is self-contained and reproducible — no S3, no
    server, just files on the shared FS. Best-effort: a snapshot failure must
    never block sharing the finding. Returns the snapshot dir name, or None.

    Isolation note: this writes into the CURRENT team's forum dir only. A fresh
    (non-seeded) team never reads another team's forum, so always snapshotting
    here does not leak code across teams — only an explicit SEED_FORUM_FROM copy
    would carry it forward.
    """
    if not idea_name or idea_name == "unknown":
        return None
    try:
        import shutil
        from aar import config as _cfg

        src = _cfg.resolve_idea_dir(idea_name)
        if not src.is_dir():
            return None
        if dest_dir.exists():
            return dest_dir.name
        shutil.copytree(src, dest_dir, ignore=_snapshot_ignore)
        return dest_dir.name
    except Exception as e:  # pragma: no cover - defensive
        print(f"[fs_forum] code snapshot skipped for {idea_name!r}: {e}")
        return None


def use_fs_forum() -> bool:
    """True when the forum should be the shared FS dir rather than an HTTP server.

    Explicit ``FORUM_BACKEND`` wins; otherwise default to the FS forum whenever
    we're in local mode (i.e. there is no central orchestrator server running).
    """
    backend = os.getenv("FORUM_BACKEND", "").strip().lower()
    if backend:
        return backend == "fs"
    return os.getenv("LOCAL_MODE", "").strip().lower() in ("1", "true", "yes")


def _findings_dir() -> Path:
    return Path(config.LOCAL_FINDINGS_DIR)


def _composite_headline(f: dict[str, Any]) -> Any:
    """Best-effort headline for ranking. Robust to the agent using either
    `headline_pct` (canonical) or `headline` (a reformatted variant we saw in
    the pilot), at top level or nested in composite_scores — so a finding never
    silently drops off the leaderboard just because of a key-name choice."""
    cs = f.get("composite_scores")
    cs = cs if isinstance(cs, dict) else {}
    for src in (f, cs):
        for key in ("headline_pct", "headline"):
            v = src.get(key)
            if isinstance(v, (int, float)):
                return v
    return None


def write_finding(payload: dict[str, Any]) -> dict[str, Any]:
    """Materialize a finding from a share_finding payload and persist it to the
    shared forum dir. Returns the stored finding dict (with id/post_id/created_at).

    Each chain writes a uniquely-named file (uuid-based), so 9 chains writing
    concurrently never collide and never overwrite each other.
    """
    finding = dict(payload)
    fid = uuid.uuid4().hex
    finding.setdefault("id", fid)
    finding.setdefault("post_id", fid[:12])
    finding.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    finding.setdefault("idea_name", payload.get("idea_name") or os.getenv("IDEA_NAME") or "unknown")

    # Snapshot the method's source code next to the finding so it's reproducible
    # without S3/server. Dir name mirrors the JSON stem: <id>_<idea>_code/.
    # Done BEFORE the JSON write so the finding records the snapshot reference.
    forum_dir = _findings_dir()
    code_dir_name = finding_filename(finding)[: -len(".json")] + "_code"
    snap = _snapshot_method_code(finding["idea_name"], forum_dir / code_dir_name)
    if snap:
        finding["code_snapshot"] = snap

    saved = save_finding_to_dir(finding, forum_dir)
    finding["_saved_path"] = str(saved) if saved else None
    return finding


def read_findings() -> list[dict[str, Any]]:
    """Load every finding JSON in the shared forum dir (all chains write here)."""
    d = _findings_dir()
    if not d.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def _rank_key(f: dict[str, Any]) -> tuple[float, float]:
    """Rank: capability-filter PASS above FAIL first, THEN by headline geo-mean
    (both descending, via reverse=True). A method that passes the 3-way capability
    filter always outranks one that fails it, regardless of geo-mean — so the
    leaderboard groups qualifying results on top, then sorts each group by geo-mean."""
    h = _composite_headline(f)
    if isinstance(h, (int, float)):
        score = float(h)
    else:
        pgr = f.get("pgr")
        score = pgr * 100.0 if isinstance(pgr, (int, float)) else float("-inf")
    cs = f.get("composite_scores")
    cs = cs if isinstance(cs, dict) else {}
    passes = 1.0 if cs.get("passes_filter") else 0.0
    return (passes, score)


def leaderboard(limit: int = 50) -> list[dict[str, Any]]:
    """Rank shared 'result' findings by headline_pct (fallback pgr*100), best first.

    Returns lightweight entries (one per peer result) for the agent to build on.
    """
    results = [
        f for f in read_findings()
        if f.get("finding_type") == "result" or _composite_headline(f) is not None
    ]
    results.sort(key=_rank_key, reverse=True)
    entries: list[dict[str, Any]] = []
    for f in results[:limit]:
        entries.append({
            "id": f.get("id"),
            "post_id": f.get("post_id"),
            "idea_name": f.get("idea_name"),
            "idea_uid": f.get("idea_uid"),
            "run_id": f.get("run_id"),
            "suite": f.get("suite"),
            "headline_pct": _composite_headline(f),
            "pgr": f.get("pgr"),
            "worked": f.get("worked"),
            "title": f.get("title"),
            "summary": f.get("summary"),
            "composite_scores": f.get("composite_scores"),
            "created_at": f.get("created_at"),
        })
    return entries
