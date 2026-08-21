#!/usr/bin/env python3
"""
Hook script for recording tool start times via Claude PreToolUse hooks.

Called before a tool executes. Records the start time so that the
PostToolUse hook can calculate duration.

Usage (from settings.json hook):
    python3 "$REPO_ROOT/aar/research_loop/hooks/log_tool_start.py"
"""

import json
import sys


def read_hook_input():
    """
    Read and parse hook input from stdin.

    Returns:
        Parsed JSON dict, or empty dict if stdin is empty/invalid
    """
    try:
        if sys.stdin.isatty():
            return {}

        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            return {}

        return json.loads(stdin_data)
    except json.JSONDecodeError as e:
        print(f"[Timing] Failed to parse stdin as JSON: {e}")
        return {}
    except Exception as e:
        print(f"[Timing] Error reading stdin: {e}")
        return {}


def main():
    """Main entry point for the hook script."""
    hook_data = read_hook_input()

    tool_name = hook_data.get("tool_name", "")
    tool_input = hook_data.get("tool_input", {})
    session_id = hook_data.get("session_id", "unknown")

    if not tool_name:
        return

    # Only track Skill and MCP tools
    if tool_name != "Skill" and not tool_name.startswith("mcp__"):
        return

    print(f"[Timing] Recording start time for: {tool_name}")

    try:
        from aar.research_loop.telemetry.tool_timing import record_start_time
        record_start_time(session_id, tool_name, tool_input)
        print(f"[Timing] Start time recorded")
    except ImportError as e:
        print(f"[Timing] Skipping (import error): {e}")
    except Exception as e:
        print(f"[Timing] Error recording start time: {e}")


if __name__ == "__main__":
    main()
