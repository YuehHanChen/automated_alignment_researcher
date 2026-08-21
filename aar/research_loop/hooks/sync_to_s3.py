#!/usr/bin/env python3
"""
Quick S3 sync script for Claude Code hooks.

Called by PostToolUse hook to sync findings and logs after each tool use.
Designed to be fast and fail silently to not block Claude.
"""

import os
from pathlib import Path


def sync():
    """Sync findings, usage stats, and logs directory to S3."""
    idea_uid = os.getenv("IDEA_UID")
    run_id = os.getenv("RUN_ID")
    if not idea_uid or not run_id:
        return

    # Skip in local mode (no S3)
    if os.getenv("LOCAL_MODE", "").lower() in ("1", "true", "yes"):
        return

    from aar.config import S3_BUCKET as s3_bucket, S3_IDEAS_PREFIX as s3_prefix, WORKSPACE_DIR as workspace_dir

    if not s3_bucket:
        return

    logs_path = Path(workspace_dir) / "aar" / "research_loop" / "logs"
    findings_path = logs_path.parent / "findings.json"
    usage_stats_path = logs_path / "usage_stats.json"

    try:
        from aar.infrastructure.s3_utils import upload_file_to_s3
    except ImportError:
        return

    s3_key_prefix = f"{s3_prefix}{idea_uid}/{run_id}/"

    # Sync findings.json
    if findings_path.exists():
        _upload_file(upload_file_to_s3, findings_path, f"{s3_key_prefix}findings.json", s3_bucket, "application/json")

    # Sync usage_stats.json (priority file - always sync)
    if usage_stats_path.exists():
        _upload_file(upload_file_to_s3, usage_stats_path, f"{s3_key_prefix}logs/usage_stats.json", s3_bucket, "application/json")

    # Sync logs directory (skip autonomous_worker.log - redundant and large)
    if not logs_path.exists():
        return

    synced_count = 0
    for log_file in logs_path.glob("*"):
        # Skip usage_stats.json as we already synced it above
        if log_file.is_file() and log_file.name != "autonomous_worker.log" and log_file.name != "usage_stats.json":
            content_type = "application/json" if log_file.suffix == ".json" else "text/plain"
            if _upload_file(upload_file_to_s3, log_file, f"{s3_key_prefix}logs/{log_file.name}", s3_bucket, content_type):
                synced_count += 1

    if synced_count > 0:
        print(f"Synced {synced_count} files from logs/")


def _upload_file(upload_fn, file_path: Path, s3_key: str, bucket: str, content_type: str) -> bool:
    """Upload a file to S3. Returns True on success."""
    try:
        upload_fn(file_path=file_path, s3_key=s3_key, bucket_name=bucket, content_type=content_type)
        print(f"Synced {file_path.name}")
        return True
    except Exception as e:
        print(f"Failed to sync {file_path.name}: {e}")
        return False


if __name__ == "__main__":
    sync()
