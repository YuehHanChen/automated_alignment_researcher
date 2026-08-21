"""
Thread-safe usage tracking for skills and MCP tools.

Tracks frequency and duration of tool calls. Persists to usage_stats.json
which gets synced to S3.

Usage:
    tracker = get_tracker()
    tracker.record("evaluate_predictions", category="mcp", duration_ms=500, success=True)
    tracker.record("thinking", category="skill", duration_ms=9000, success=True)
    stats = tracker.get_stats()
"""

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


_tracker_instance: Optional["UsageTracker"] = None
_tracker_lock = threading.Lock()


def get_tracker(log_dir: Optional[Path] = None) -> "UsageTracker":
    """Get or create the global UsageTracker instance."""
    global _tracker_instance

    with _tracker_lock:
        if _tracker_instance is None:
            if log_dir is None:
                from aar.config import WORKSPACE_DIR
                logs_dir = os.getenv("LOGS_DIR", f"{WORKSPACE_DIR}/aar/research_loop/logs")
                log_dir = Path(logs_dir)

            log_dir.mkdir(parents=True, exist_ok=True)
            _tracker_instance = UsageTracker(log_file=str(log_dir / "usage_stats.json"))

        return _tracker_instance


def reset_tracker():
    """Reset the global tracker instance."""
    global _tracker_instance
    with _tracker_lock:
        _tracker_instance = None


class UsageTracker:
    """Thread-safe usage tracker. Persists statistics to a JSON file."""

    def __init__(self, log_file: str = "usage_stats.json"):
        self.log_file = Path(log_file)
        self._lock = threading.Lock()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._stats = self._load_or_create()

    def _load_or_create(self) -> Dict[str, Any]:
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    stats = json.load(f)
                stats["last_session_id"] = self.session_id
                stats["last_updated"] = datetime.now().isoformat()
                return stats
            except (json.JSONDecodeError, IOError):
                pass

        return {
            "session_id": self.session_id,
            "session_start": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "skills": {},
            "mcp_tools": {},
            "summary": {
                "total_skill_calls": 0,
                "total_mcp_calls": 0,
                "total_api_calls": 0,
            },
        }

    def record(
        self,
        name: str,
        category: str = "mcp",
        duration_ms: Optional[int] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a tool/skill invocation.

        Args:
            name: Tool or skill name
            category: "skill" or "mcp"
            duration_ms: Duration in milliseconds
            success: Whether the call succeeded
            metadata: Optional additional metadata
        """
        bucket = "skills" if category == "skill" else "mcp_tools"
        summary_key = "total_skill_calls" if category == "skill" else "total_mcp_calls"

        with self._lock:
            ts = datetime.now().isoformat()

            if name not in self._stats[bucket]:
                self._stats[bucket][name] = {
                    "count": 0, "total_duration_ms": 0,
                    "successes": 0, "failures": 0,
                    "last_used": None, "calls": [],
                }

            entry = self._stats[bucket][name]
            entry["count"] += 1
            entry["last_used"] = ts
            entry["successes" if success else "failures"] += 1
            if duration_ms is not None:
                entry["total_duration_ms"] += duration_ms

            call = {"timestamp": ts, "duration_ms": duration_ms, "success": success}
            if metadata:
                call["metadata"] = metadata
            entry["calls"].append(call)
            if len(entry["calls"]) > 100:
                entry["calls"] = entry["calls"][-100:]

            self._stats["summary"][summary_key] += 1
            self._stats["summary"]["total_api_calls"] += 1
            self._stats["last_updated"] = ts
            self._save()

    # Backward-compatible aliases
    def record_skill(self, skill_name, **kwargs):
        self.record(skill_name, category="skill", **kwargs)

    def record_mcp_tool(self, tool_name, **kwargs):
        self.record(tool_name, category="mcp", **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get a copy of current statistics."""
        with self._lock:
            return json.loads(json.dumps(self._stats))

    def _save(self) -> None:
        """Persist stats to file. Must be called with lock held."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            temp = self.log_file.with_suffix('.json.tmp')
            with open(temp, 'w') as f:
                json.dump(self._stats, f, indent=2)
            temp.replace(self.log_file)
        except Exception as e:
            print(f"[UsageTracker] Failed to save: {e}")
