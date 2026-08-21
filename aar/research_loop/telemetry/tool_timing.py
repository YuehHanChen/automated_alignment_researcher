#!/usr/bin/env python3
"""
Simple file-based timing tracker for tool invocations.

Uses a temp file to track start times of tool calls, keyed by
session_id + tool_name + tool_input hash. This allows PreToolUse
hooks to record start time and PostToolUse hooks to calculate duration.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional


def _get_timing_file() -> Path:
    """Get the path to the timing state file."""
    from aar.config import WORKSPACE_DIR
    logs_dir = os.getenv("LOGS_DIR", f"{WORKSPACE_DIR}/aar/research_loop/logs")
    return Path(logs_dir) / ".tool_timing.json"


def _get_tool_key(session_id: str, tool_name: str, tool_input: dict) -> str:
    """
    Generate a unique key for a tool invocation.

    Uses session_id + tool_name + hash of tool_input to identify
    matching PreToolUse and PostToolUse events.
    """
    input_str = json.dumps(tool_input, sort_keys=True) if tool_input else ""
    input_hash = hashlib.md5(input_str.encode()).hexdigest()[:8]
    return f"{session_id}:{tool_name}:{input_hash}"


def record_start_time(session_id: str, tool_name: str, tool_input: dict) -> None:
    """
    Record the start time of a tool invocation (called from PreToolUse hook).

    Args:
        session_id: Claude session ID
        tool_name: Name of the tool being invoked
        tool_input: Tool input parameters
    """
    timing_file = _get_timing_file()
    timing_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing timing data
    timing_data = {}
    if timing_file.exists():
        try:
            with open(timing_file, 'r') as f:
                timing_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            timing_data = {}

    # Add new entry
    key = _get_tool_key(session_id, tool_name, tool_input)
    timing_data[key] = {
        "start_time": time.time(),
        "session_id": session_id,
        "tool_name": tool_name,
    }

    # Clean up old entries (older than 1 hour)
    cutoff = time.time() - 3600
    timing_data = {k: v for k, v in timing_data.items()
                   if v.get("start_time", 0) > cutoff}

    # Save
    try:
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f)
    except IOError as e:
        print(f"[Timing] Failed to save timing data: {e}")


def get_duration_ms(session_id: str, tool_name: str, tool_input: dict) -> Optional[int]:
    """
    Calculate duration in milliseconds for a tool invocation (called from PostToolUse hook).

    Args:
        session_id: Claude session ID
        tool_name: Name of the tool being invoked
        tool_input: Tool input parameters

    Returns:
        Duration in milliseconds, or None if start time not found
    """
    timing_file = _get_timing_file()
    if not timing_file.exists():
        return None

    try:
        with open(timing_file, 'r') as f:
            timing_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    key = _get_tool_key(session_id, tool_name, tool_input)
    entry = timing_data.get(key)

    if not entry or "start_time" not in entry:
        return None

    duration_ms = int((time.time() - entry["start_time"]) * 1000)

    # Remove the entry (it's been consumed)
    del timing_data[key]
    try:
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f)
    except IOError:
        pass

    return duration_ms
