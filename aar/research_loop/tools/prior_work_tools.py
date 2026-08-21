"""
Prior Work Tools - Snapshot system for preserving and sharing research progress.

Provides tools for agents to download specific snapshots to build upon.
Snapshots are automatically created via share_finding (finding_type='result').
"""

import os
from pathlib import Path
from typing import Any, Dict

from claude_agent_sdk import tool, create_sdk_mcp_server

from .http_utils import (
    get_server_url,
    validate_safe_identifier,
    validate_safe_path,
    HAS_HTTPX,
)

if HAS_HTTPX:
    import httpx
else:
    import requests


@tool(
    "download_snapshot",
    "Download a specific snapshot to local filesystem.",
    {
        "snapshot_id": str,
        "target_dir": str,
    }
)
async def download_snapshot(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download and extract a workspace snapshot.

    Args:
        snapshot_id: The snapshot ID to download (16-char hash)
        target_dir: Where to extract files (default: /workspace/prior_work/snapshots/{snapshot_id})

    Returns:
        Dict with local_path and files_count
    """
    try:
        commit_id = args.get("snapshot_id", "").strip() or args.get("commit_id", "").strip()
        if not commit_id:
            return {
                "content": [{
                    "type": "text",
                    "text": "ERROR: snapshot_id is required."
                }]
            }

        try:
            commit_id = validate_safe_identifier(commit_id, "snapshot_id")
        except ValueError as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"ERROR: {str(e)}"
                }]
            }

        api_url = get_server_url()

        if HAS_HTTPX:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{api_url}/api/snapshots/{commit_id}")
                response.raise_for_status()
                commit_data = response.json()
        else:
            response = requests.get(f"{api_url}/api/snapshots/{commit_id}", timeout=30)
            response.raise_for_status()
            commit_data = response.json()

        if "error" in commit_data:
            return {
                "content": [{
                    "type": "text",
                    "text": f"ERROR: {commit_data['error']}"
                }]
            }

        idea_uid = commit_data.get("idea_uid")
        run_id = commit_data.get("run_id")

        if not idea_uid or not run_id:
            return {
                "content": [{
                    "type": "text",
                    "text": "ERROR: Could not determine idea_uid and run_id for this snapshot."
                }]
            }

        target_dir = args.get("target_dir", "").strip()
        if not target_dir:
            target_dir = f"/workspace/prior_work/snapshots/{commit_id}"

        try:
            target_dir = validate_safe_path(target_dir, "target_dir", "/workspace")
        except ValueError as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"ERROR: {str(e)}"
                }]
            }

        target_path = Path(target_dir)

        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key or not aws_secret_key:
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        f"NOTE: AWS credentials not available for S3 download.\n\n"
                        f"Snapshot info from API:\n"
                        f"- Snapshot ID: {commit_id}\n"
                        f"- Title: {commit_data.get('title', 'N/A')}\n"
                        f"- Idea: {commit_data.get('idea_name', 'N/A')}\n"
                        f"- PGR: {commit_data.get('pgr', 'N/A')}\n"
                        f"- Files: {commit_data.get('files_snapshot', [])}\n\n"
                        f"To download, set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
                    )
                }]
            }

        from aar.config import S3_BUCKET as bucket, S3_IDEAS_PREFIX as prefix

        from aar.infrastructure.s3_utils import download_snapshot_from_s3

        metadata, extracted_files = download_snapshot_from_s3(
            commit_id=commit_id,
            idea_uid=idea_uid,
            run_id=run_id,
            target_dir=target_path,
            bucket_name=bucket,
            prefix=prefix,
        )

        output_lines = [
            f"Downloaded snapshot {commit_id[:8]}...",
            f"  Message: {metadata.get('message', 'N/A')}",
            f"  Idea: {metadata.get('idea_name', 'N/A')}",
            f"  PGR: {metadata.get('metrics', {}).get('pgr', 'N/A')}",
            f"  Extracted {len(extracted_files)} files to {target_path}",
            "",
            "Key files:",
        ]

        py_files = [f for f in extracted_files if f.endswith('.py')]
        for f in py_files[:10]:
            output_lines.append(f"  - {f}")
        if len(py_files) > 10:
            output_lines.append(f"  ... and {len(py_files) - 10} more Python files")

        output_lines.append(f"\nYou can now reference the code at {target_path}")

        return {
            "content": [{
                "type": "text",
                "text": "\n".join(output_lines)
            }]
        }

    except FileNotFoundError as e:
        return {
            "content": [{
                "type": "text",
                "text": f"ERROR: Snapshot not found in S3: {str(e)}"
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"ERROR: Failed to download snapshot: {str(e)}"
            }]
        }


def create_prior_work_tools_server():
    """Create MCP server with snapshot tools."""
    return create_sdk_mcp_server(
        name="prior-work-tools",
        version="3.0.0",
        tools=[
            download_snapshot,
        ]
    )


