"""
Background polling sync for shared findings.

Polls the server's GET /api/findings/all endpoint periodically,
deduplicates by finding ID, and saves new findings as individual
JSON files in a local directory. Agents search these files directly
using Read/Glob/Grep — no server round-trip needed.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from .http_utils import get_server_url, DEFAULT_HEADERS, HAS_HTTPX

# Use requests (synchronous) since this runs in a background thread
try:
    import requests
except ImportError:
    requests = None

if HAS_HTTPX:
    import httpx


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename component."""
    if not name:
        return "unknown"
    sanitized = ""
    for c in name:
        if c.isalnum() or c in "-_":
            sanitized += c
        else:
            sanitized += "_"
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_")[:100]


def finding_filename(finding: dict) -> str:
    """Build the canonical filename for a finding.

    Uses {id}_{sanitized_idea_name}.json so that both immediate save
    (in share_finding) and background sync produce identical filenames,
    preventing duplicates.
    """
    finding_id = finding.get("id")
    idea_name = finding.get("idea_name") or "unknown"
    name_part = _sanitize_filename(idea_name)
    return f"{finding_id}_{name_part}.json"


def save_finding_to_dir(finding: dict, findings_dir: Path) -> Optional[Path]:
    """Save a single finding dict as a JSON file.

    Returns the file path if written, None if already exists or failed.
    """
    findings_dir.mkdir(parents=True, exist_ok=True)
    filename = finding_filename(finding)
    filepath = findings_dir / filename

    if filepath.exists():
        return None

    try:
        # Write atomically via temp file to avoid partial reads
        tmp = filepath.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(finding, f, indent=2, default=str)
        tmp.rename(filepath)
        return filepath
    except Exception as e:
        print(f"[findings-sync] Failed to write {filepath}: {e}")
        # Clean up temp file if rename failed
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return None


class FindingsSync:
    """Background thread that polls the server for shared findings.

    Usage:
        sync = FindingsSync(findings_dir=Path("shared_findings"))
        sync.start()
        # ... agent runs ...
        sync.stop()
    """

    def __init__(
        self,
        findings_dir: Path,
        poll_interval: int = 60,
        server_url: Optional[str] = None,
    ):
        self.findings_dir = Path(findings_dir)
        self.poll_interval = poll_interval
        self.server_url = server_url  # Resolved lazily if None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _get_server_url(self) -> str:
        if self.server_url:
            return self.server_url.rstrip("/")
        return get_server_url()

    def _fetch_all_findings(self) -> list:
        """GET /api/findings/all from the server."""
        url = f"{self._get_server_url()}/api/findings/all"
        headers = DEFAULT_HEADERS

        if requests is not None:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json().get("findings", [])
        elif HAS_HTTPX:
            # Synchronous httpx client as fallback
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.json().get("findings", [])
        else:
            raise RuntimeError("Neither requests nor httpx available")

    def sync_once(self) -> int:
        """Run one sync cycle. Returns number of new findings saved."""
        try:
            findings = self._fetch_all_findings()
        except Exception as e:
            print(f"[findings-sync] Fetch failed: {e}")
            return 0

        new_count = 0
        for finding in findings:
            if not finding.get("id"):
                continue
            path = save_finding_to_dir(finding, self.findings_dir)
            if path:
                new_count += 1

        if new_count > 0:
            print(f"[findings-sync] Saved {new_count} new finding(s)")

        return new_count

    def _run(self):
        """Background thread loop."""
        # Initial sync immediately
        self.sync_once()

        while not self._stop_event.is_set():
            self._stop_event.wait(self.poll_interval)
            if self._stop_event.is_set():
                break
            self.sync_once()

    def start(self):
        """Start the background sync thread."""
        if self._thread and self._thread.is_alive():
            return

        self.findings_dir.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="findings-sync",
            daemon=True,
        )
        self._thread.start()
        print(f"[findings-sync] Started (dir={self.findings_dir}, interval={self.poll_interval}s)")

    def stop(self):
        """Stop the background sync thread and do a final sync."""
        if not self._thread:
            return

        self._stop_event.set()
        self._thread.join(timeout=10)
        self._thread = None

        # Final sync to catch anything missed
        self.sync_once()
        print("[findings-sync] Stopped")
