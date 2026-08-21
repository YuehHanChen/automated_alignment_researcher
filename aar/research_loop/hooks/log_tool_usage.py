#!/usr/bin/env python3
"""
Hook script for logging tool usage via Claude PostToolUse hooks.

Handles both Skill tools and MCP tools. Claude Code hooks receive
data via stdin as JSON.

For Skills:
- tool_name: "Skill"
- tool_input: {"skill": "skill-name", "args": "..."}

For MCP tools:
- tool_name: "mcp__server__tool" (e.g., "mcp__prior-work-tools__download_snapshot")
- tool_input: {tool-specific args}

Duration tracking:
- Uses tool_timing module to get start time recorded by PreToolUse hook
- Calculates duration_ms = end_time - start_time

Usage (from settings.json hook):
    python3 "$REPO_ROOT/aar/research_loop/hooks/log_tool_usage.py"
"""

import json
import os
import sys


def get_duration_ms(session_id: str, tool_name: str, tool_input: dict):
    """Get duration in ms from the tool timing tracker."""
    try:
        from aar.research_loop.telemetry.tool_timing import get_duration_ms as _get_duration
        return _get_duration(session_id, tool_name, tool_input)
    except ImportError:
        return None
    except Exception as e:
        print(f"[Usage] Error getting duration: {e}")
        return None


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
        print(f"[Usage] Failed to parse stdin as JSON: {e}")
        return {}
    except Exception as e:
        print(f"[Usage] Error reading stdin: {e}")
        return {}


def extract_mcp_query_params(mcp_tool_name: str, tool_input: dict) -> dict:
    """
    Extract query-relevant parameters from specific MCP tools for logging.

    Returns a dict of query params to include in metadata, or empty dict.
    """
    if not isinstance(tool_input, dict):
        return {}

    if mcp_tool_name == "query_findings":
        params = {}
        for key in ("query", "limit"):
            if key in tool_input:
                params[key] = tool_input[key]
        return params

    return {}


def extract_mcp_response_results(mcp_tool_name: str, tool_response) -> list:
    """
    Extract response results from specific MCP tools for logging.

    For query_findings, extracts a summary of each returned finding
    (idea_name, title, pgr, finding_type, relevance) for display in the UI.

    Returns a list of result summaries, or empty list.
    """
    if mcp_tool_name != "query_findings":
        return []

    if not tool_response or not isinstance(tool_response, str):
        return []

    try:
        response_data = json.loads(tool_response)

        results = response_data.get("results", [])
        if not results:
            return []

        # Extract summary of each finding for UI display
        summaries = []
        for r in results:
            summary = {}
            # Core fields for collapsed view
            if r.get("idea_name"):
                summary["idea_name"] = r["idea_name"]
            if r.get("title"):
                summary["title"] = r["title"]
            if r.get("pgr") is not None:
                summary["pgr"] = r["pgr"]
            if r.get("finding_type"):
                summary["finding_type"] = r["finding_type"]
            if r.get("_relevance"):
                summary["relevance"] = r["_relevance"]
            if r.get("worked") is not None:
                summary["worked"] = r["worked"]
            # Include content for expandable view (truncate to 500 chars)
            if r.get("content"):
                content_text = r["content"]
                summary["content"] = content_text[:500] + ("..." if len(content_text) > 500 else "")
            if summary:
                summaries.append(summary)

        return summaries

    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        return []


def parse_mcp_tool_name(tool_name: str) -> tuple:
    """
    Parse MCP tool name from format: mcp__server__tool

    Returns:
        (server_name, tool_name) or (None, None) if not an MCP tool
    """
    if not tool_name or not tool_name.startswith("mcp__"):
        return None, None

    parts = tool_name.split("__")
    if len(parts) >= 3:
        server = parts[1]
        tool = "__".join(parts[2:])  # Handle tools with __ in name
        return server, tool

    return None, None


def main():
    """Main entry point for the hook script."""
    hook_data = read_hook_input()

    tool_name = hook_data.get("tool_name", "")
    tool_input = hook_data.get("tool_input", {})
    tool_response = hook_data.get("tool_response", {})
    session_id = hook_data.get("session_id")

    if not tool_name:
        print("[Usage] No tool_name in hook data, skipping")
        return

    print(f"[Usage] Processing tool: {tool_name}")

    try:
        from aar.research_loop.telemetry.usage_tracker import get_tracker
        tracker = get_tracker()

        # Get duration from the timing tracker (if PreToolUse recorded start time)
        duration_ms = get_duration_ms(session_id, tool_name, tool_input)
        if duration_ms is not None:
            print(f"[Usage] Duration: {duration_ms}ms")

        # Handle Skill tool
        if tool_name == "Skill":
            if isinstance(tool_input, dict):
                skill_name = tool_input.get("skill")
                skill_args = tool_input.get("args")

                if skill_name:
                    tracker.record_skill(
                        skill_name=skill_name,
                        duration_ms=duration_ms,
                        success=True,
                        metadata={
                            "args": skill_args,
                            "session_id": session_id,
                        } if skill_args or session_id else None,
                    )
                    print(f"[Usage] Recorded skill: {skill_name}")
                else:
                    print(f"[Usage] Skill tool but no skill name in input")

        # Handle MCP tools
        elif tool_name.startswith("mcp__"):
            server_name, mcp_tool_name = parse_mcp_tool_name(tool_name)

            if mcp_tool_name:
                # Check if tool_response indicates success
                success = True
                if isinstance(tool_response, dict):
                    # MCP tools often return {"content": [...]} on success
                    # or {"error": ...} on failure
                    if "error" in tool_response:
                        success = False

                metadata = {
                    "server": server_name,
                    "input_keys": list(tool_input.keys()) if isinstance(tool_input, dict) else None,
                    "session_id": session_id,
                }

                # Add query params for discovery tools
                query_params = extract_mcp_query_params(mcp_tool_name, tool_input)
                if query_params:
                    metadata["query_params"] = query_params

                # Add response results for query_findings (for UI display)
                response_results = extract_mcp_response_results(mcp_tool_name, tool_response)
                if response_results:
                    metadata["response_results"] = response_results

                tracker.record_mcp_tool(
                    tool_name=mcp_tool_name,
                    duration_ms=duration_ms,
                    success=success,
                    metadata=metadata,
                )
                print(f"[Usage] Recorded MCP tool: {mcp_tool_name} (server: {server_name})")
            else:
                print(f"[Usage] Could not parse MCP tool name from: {tool_name}")

        else:
            # Other built-in tools - we could track these too if needed
            print(f"[Usage] Skipping built-in tool: {tool_name}")

    except ImportError as e:
        print(f"[Usage] Skipping (import error): {e}")
    except Exception as e:
        print(f"[Usage] Error recording usage: {e}")


if __name__ == "__main__":
    main()
