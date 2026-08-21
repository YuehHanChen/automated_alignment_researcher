#!/usr/bin/env python3
"""
Execute autonomous research loop - runs on spawned RunPod pod.

This script is the entrypoint for autonomous research pods. It:
1. Downloads dataset, cache, and results from S3
2. Downloads the selected idea from S3
3. Launches the autonomous agent loop using Claude Agent SDK
4. Claude continuously iterates and improves on the selected idea for up to 5 days
5. Results are uploaded to S3 after each iteration

Usage:
    python -m aar.infrastructure.execute_autonomous <idea_uid>

    idea_uid is required - Claude will iterate on this specific idea.

Environment variables (required):
    S3_BUCKET            - S3 bucket for results
    S3_ENDPOINT_URL      - S3 endpoint (RunPod)
    AWS_ACCESS_KEY_ID    - AWS credentials
    AWS_SECRET_ACCESS_KEY - AWS credentials
    ANTHROPIC_API_KEY    - Required for Claude Agent SDK

Environment variables (optional):
    DATASET_NAME         - Dataset to use (default: math-claudefilter-imbalance)
    WEAK_MODEL           - Weak supervisor model
    STRONG_MODEL         - Strong student model
    IDEA_UID             - UID of idea to iterate on (alternative to positional arg)
    IDEA_NAME            - Name of the idea (for logging)
    FULL_AUTO_MAX_RUNTIME_SECONDS - Max runtime in seconds (default: 5 days)
"""
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Add workspace to path
workspace = Path(os.getenv("WORKSPACE_DIR", "/workspace"))
if str(workspace) not in sys.path:
    sys.path.insert(0, str(workspace))

print(f"✓ Using workspace: {workspace}")

# Import config for log directory
try:
    from aar.config import LOGS_DIR
    logs_dir = Path(LOGS_DIR)
except ImportError:
    logs_dir = workspace / "aar" / "research_loop" / "logs"

# Create logs directory if it doesn't exist
logs_dir.mkdir(parents=True, exist_ok=True)

# Set up logging
log_file = logs_dir / "autonomous_worker.log"


class TeeOutput:
    """Write to both file and stdout/stderr"""
    def __init__(self, file_path, stream):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        if self.file and not self.file.closed:
            self.file.flush()
            self.file.close()


try:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeOutput(log_file, original_stdout)
    sys.stderr = TeeOutput(log_file, original_stderr)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(original_stdout)],
        force=True,
    )
    print(f"✓ Logging initialized - log file: {log_file}")
except Exception as e:
    print(f"⚠️  Warning: Failed to create log file: {e}")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

logger = logging.getLogger(__name__)


def download_idea(idea_uid: str) -> dict | None:
    """Download idea from S3 and save locally for the prompt to reference.
    
    Args:
        idea_uid: The UID of the idea to download
        
    Returns:
        The idea dict if successful, None otherwise
    """
    try:
        from aar.infrastructure.s3_utils import get_s3_client
    except ImportError as e:
        print(f"❌ ERROR: Failed to import s3_utils: {e}")
        return None

    from aar.config import S3_BUCKET as s3_bucket, S3_IDEAS_PREFIX as s3_ideas_prefix

    print(f"\n{'='*80}")
    print(f"Downloading idea: {idea_uid}")
    print(f"{'='*80}")

    try:
        s3_client = get_s3_client()
        idea_key = f"{s3_ideas_prefix}{idea_uid}/idea.json"
        
        print(f"  S3 path: s3://{s3_bucket}/{idea_key}")
        
        response = s3_client.get_object(Bucket=s3_bucket, Key=idea_key)
        idea_data = json.loads(response['Body'].read().decode('utf-8'))
        
        # Save idea locally for the prompt to reference
        idea_dir = workspace / "aar" / "research_loop" / "target_idea"
        idea_dir.mkdir(parents=True, exist_ok=True)
        
        idea_file = idea_dir / "idea.json"
        with open(idea_file, 'w') as f:
            json.dump(idea_data, f, indent=2)
        
        print(f"✓ Idea downloaded and saved to: {idea_file}")
        print(f"  Name: {idea_data.get('Name', 'unknown')}")
        return idea_data
        
    except Exception as e:
        print(f"❌ Error downloading idea: {e}")
        return None


def download_prerequisites():
    """Download dataset, cache, and results from S3."""
    try:
        from aar.infrastructure.s3_utils import download_s3_directory
    except ImportError as e:
        print(f"❌ ERROR: Failed to import s3_utils: {e}")
        return False

    from aar.config import (
        S3_BUCKET as s3_bucket, S3_ENDPOINT_URL as s3_endpoint,
        S3_RESULTS_PREFIX, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    )

    print(f"\n{'='*80}")
    print(f"S3 Configuration: bucket={s3_bucket}, endpoint={s3_endpoint}")
    print(f"{'='*80}\n")

    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print(f"❌ ERROR: AWS credentials not set.")
        return False

    if s3_endpoint:
        os.environ["S3_ENDPOINT_URL"] = s3_endpoint

    from aar.config import DATASET_NAME as dataset_name, DATA_DIR

    # Download dataset
    data_dir = DATA_DIR
    local_data_dir = Path(data_dir)

    print(f"\n{'='*80}")
    print(f"Downloading dataset: {dataset_name}")
    print(f"{'='*80}")

    # Check if dataset exists locally
    required_files = ["train_unlabel.jsonl", "test.jsonl"]
    from aar.config import AAR_MODE as aar_mode
    if not aar_mode:
        required_files.append("train_label.jsonl")

    dataset_exists = all((local_data_dir / f).exists() for f in required_files)

    if not dataset_exists:
        s3_data_prefix = "data/"
        try:
            success = download_s3_directory(
                local_dir=local_data_dir,
                bucket_name=s3_bucket,
                s3_prefix=f"{s3_data_prefix}{dataset_name}/",
                description=f"dataset '{dataset_name}'",
            )
            if not success:
                print(f"❌ Failed to download dataset from S3")
                return False
            print(f"✓ Dataset downloaded to {local_data_dir}")
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False
    else:
        print(f"✓ Dataset already exists at {local_data_dir}")

    # Download cache
    cache_dir = workspace / "cache_results"
    print(f"\n{'='*80}")
    print(f"Downloading cache files")
    print(f"{'='*80}")

    try:
        cache_downloaded = download_s3_directory(
            local_dir=cache_dir,
            bucket_name=s3_bucket,
            s3_prefix="cache_results/",
            description="cache files",
        )
        if cache_downloaded:
            print(f"✓ Cache downloaded to {cache_dir}")
        else:
            print(f"⚠️  No cache found in S3 (OK for first run)")
    except Exception as e:
        print(f"⚠️  Error downloading cache: {e}")
        print(f"   Continuing without cache...")

    # Download results directory from S3 if not present locally
    # This allows resuming experiments or accessing previous results
    results_dir = workspace / "results"
    print(f"\n{'='*80}")
    print(f"Downloading result files")
    print(f"{'='*80}")

    try:
        results_downloaded = download_s3_directory(
            local_dir=results_dir,
            bucket_name=s3_bucket,
            s3_prefix=S3_RESULTS_PREFIX,
            description="result files",
        )
        if results_downloaded:
            print(f"✓ Results downloaded to {results_dir}")
        else:
            print(f"⚠️  No results found in S3 (OK for first run)")
    except Exception as e:
        print(f"⚠️  Error downloading results: {e}")
        print(f"   Continuing without existing results...")

    # Set DATA_DIR environment variable for the autonomous loop
    os.environ["DATA_DIR"] = str(local_data_dir)

    return True


def launch_autonomous_loop(idea_uid: str, idea_name: str) -> bool:
    """
    Launch autonomous agent loop using Claude Agent SDK.

    Args:
        idea_uid: Required S3 idea UID to iterate on
        idea_name: Name of the idea

    Returns:
        True if successful, False otherwise
    """
    import asyncio

    if not idea_uid:
        print("❌ ERROR: idea_uid is required")
        return False

    from aar.config import FULL_AUTO_MAX_RUNTIME_SECONDS
    max_runtime_seconds = FULL_AUTO_MAX_RUNTIME_SECONDS

    print(f"\n{'='*80}")
    print(f"Launching autonomous research loop")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Max runtime: {max_runtime_seconds}s ({max_runtime_seconds / 3600:.1f} hours)")
    print(f"Idea UID: {idea_uid}")
    print(f"Idea Name: {idea_name}")
    print(f"{'='*80}\n")

    try:
        from aar.research_loop.agent import AutonomousAgentLoop

        loop = AutonomousAgentLoop(
            idea_uid=idea_uid,
            idea_name=idea_name,
            workspace=workspace,
            max_runtime_seconds=max_runtime_seconds,
        )

        result = asyncio.run(loop.run())

        success = result.get("stop_reason") != "unrecoverable_error"
        print(f"\n✓ Autonomous loop completed: {result.get('stop_reason', 'unknown')}")
        print(f"  Sessions: {result.get('sessions', 0)}")
        print(f"  Duration: {result.get('duration_seconds', 0)/3600:.1f}h")

        return success

    except KeyboardInterrupt:
        print(f"\n⚠️  Autonomous loop interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running autonomous loop: {e}")
        import traceback
        traceback.print_exc()
        return False



def upload_final_artifacts(idea_uid: str, idea_name: str, success: bool):
    """
    Upload final artifacts to S3 at end of autonomous run.

    IMPORTANT: Upload order matters to avoid race conditions with server!
    Upload workspace.tar.gz and logs FIRST, then results.json LAST.
    Server detects completion by checking for results.json.

    Args:
        idea_uid: The S3 idea UID
        idea_name: Name of the idea
        success: Whether the run was successful
    """
    try:
        from aar.infrastructure.s3_utils import (
            upload_file_to_s3,
            upload_commit_to_s3,
            generate_commit_id,
        )
    except ImportError as e:
        print(f"⚠️  Failed to import s3_utils: {e}")
        return

    from aar.config import S3_BUCKET as s3_bucket, S3_IDEAS_PREFIX as s3_ideas_prefix, DATASET_NAME as dataset_name, WEAK_MODEL, STRONG_MODEL
    run_id = os.getenv("RUN_ID", f"auto-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    print(f"\n{'='*80}")
    print(f"Uploading final artifacts to S3")
    print(f"{'='*80}")

    # Load findings to get final metrics
    # Note: findings.json is in logs_dir.parent (e.g., research_loop/findings.json)
    # while session logs are in logs_dir (e.g., research_loop/logs/)
    findings_path = logs_dir.parent / "findings.json"
    findings_data = {}
    if findings_path.exists():
        try:
            with open(findings_path) as f:
                findings_data = json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to read findings: {e}")

    # Extract metrics from findings
    metrics = {}
    if findings_data.get("iterations"):
        best_iter_num = findings_data.get("best_iteration", 0)
        for it in findings_data.get("iterations", []):
            if it.get("iteration") == best_iter_num:
                metrics = {
                    "pgr": it.get("pgr"),
                    "transfer_acc": it.get("transfer_acc"),
                    "weak_acc": it.get("weak_acc"),
                    "strong_acc": it.get("strong_acc"),
                }
                break
    metrics["best_pgr"] = findings_data.get("best_pgr", 0.0)

    # =========================================================================
    # STEP 1: Upload workspace as final commit (before results.json)
    # =========================================================================
    commit_id = None
    try:
        print(f"[1/4] Creating final commit...")

        # Generate commit ID
        timestamp = datetime.now().isoformat()
        commit_id = generate_commit_id(
            experiment_id=0,
            sequence_number=len(findings_data.get("iterations", [])),
            message=f"Final commit: {idea_name}",
            timestamp=timestamp,
        )

        s3_key, archive_size, files_list = upload_commit_to_s3(
            idea_uid=idea_uid,
            run_id=run_id,
            commit_id=commit_id,
            workspace_dir=workspace,
            metadata={
                "message": f"Final commit: {idea_name}",
                "metrics": metrics,
                "idea_name": idea_name,
                "status": "completed" if success else "failed",
                "total_iterations": len(findings_data.get("iterations", [])),
            },
            bucket_name=s3_bucket,
            prefix=s3_ideas_prefix,
        )
        print(f"✓ Created final commit {commit_id}")
        print(f"  Files: {len(files_list)}, Size: {archive_size / 1024:.1f} KB")

        # Register with orchestrator API via /api/findings/share
        orchestrator_url = os.getenv("ORCHESTRATOR_API_URL")
        if orchestrator_url:
            try:
                import requests
                requests.post(
                    f"{orchestrator_url.rstrip('/')}/api/findings/share",
                    json={
                        "summary": f"Final snapshot: {idea_name}",
                        "title": f"[Snapshot] Final commit for {idea_name}",
                        "idea_uid": idea_uid,
                        "idea_name": idea_name,
                        "run_id": run_id,
                        "finding_type": "result",
                        "pgr": metrics.get("pgr") if isinstance(metrics, dict) else None,
                        "transfer_acc": metrics.get("transfer_acc") if isinstance(metrics, dict) else None,
                        "commit_id": commit_id,
                        "s3_key": s3_key,
                        "s3_path": f"s3://{s3_bucket}/{s3_key}",
                        "sequence_number": len(findings_data.get("iterations", [])),
                        "files_snapshot": files_list,
                        "dataset": dataset_name,
                        "weak_model": WEAK_MODEL,
                        "strong_model": STRONG_MODEL,
                    },
                    timeout=30,
                )
                print(f"✓ Registered commit with orchestrator")
            except Exception as e:
                print(f"⚠️  Failed to register commit: {e}")

    except Exception as e:
        print(f"⚠️  Failed to create final commit: {e}")

    # =========================================================================
    # STEP 2: Upload findings.json as worker_pod.log
    # =========================================================================
    try:
        print(f"[2/4] Uploading findings as worker_pod.log...")
        worker_log_file = logs_dir / "worker_pod.log"
        with open(worker_log_file, "w") as f:
            json.dump(findings_data, f, indent=2)

        worker_log_key = f"{s3_ideas_prefix}{idea_uid}/{run_id}/worker_pod.log"
        upload_file_to_s3(
            file_path=worker_log_file,
            s3_key=worker_log_key,
            bucket_name=s3_bucket,
            content_type="text/plain",
        )
        print(f"✓ Uploaded worker_pod.log to s3://{s3_bucket}/{worker_log_key}")
    except Exception as e:
        print(f"⚠️  Failed to upload worker_pod.log: {e}")

    # =========================================================================
    # STEP 3: Upload session logs
    # =========================================================================
    try:
        print(f"[3/4] Uploading session logs...")
        session_logs = list(logs_dir.glob("session_*.log"))
        for log_file in session_logs:
            log_key = f"{s3_ideas_prefix}{idea_uid}/{run_id}/logs/{log_file.name}"
            upload_file_to_s3(
                file_path=log_file,
                s3_key=log_key,
                bucket_name=s3_bucket,
                content_type="text/plain",
            )
        print(f"✓ Uploaded {len(session_logs)} session logs")
    except Exception as e:
        print(f"⚠️  Failed to upload session logs: {e}")

    # =========================================================================
    # STEP 4: Upload results.json LAST (signals completion to server)
    # =========================================================================
    # Build results.json (compatible with normal mode format)
    # This is the file WebUI checks to detect completion
    results = {
        "idea_name": idea_name,
        "idea_uid": idea_uid,
        "run_id": run_id,
        "status": "completed" if success else "failed",
        "mode": "autonomous",
        "pgr": findings_data.get("best_pgr", 0.0),
        "best_iteration": findings_data.get("best_iteration", 0),
        "total_iterations": len(findings_data.get("iterations", [])),
        "transfer_acc": metrics.get("transfer_acc"),
        "weak_acc": metrics.get("weak_acc"),
        "strong_acc": metrics.get("strong_acc"),
        "completed_at": datetime.now().isoformat(),
        "final_commit_id": commit_id,
    }

    # Upload results.json LAST
    print(f"[4/4] Uploading results.json (completion signal)...")
    results_file = logs_dir / "results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        results_key = f"{s3_ideas_prefix}{idea_uid}/{run_id}/results.json"
        upload_file_to_s3(
            file_path=results_file,
            s3_key=results_key,
            bucket_name=s3_bucket,
            content_type="application/json",
        )
        print(f"✓ Uploaded results.json to s3://{s3_bucket}/{results_key}")
        print(f"  (Server will detect this as completion signal)")
    except Exception as e:
        print(f"⚠️  Failed to upload results.json: {e}")


def main():
    """Main entrypoint for autonomous worker pod."""
    print(f"\n{'='*80}")
    print(f"Autonomous Research Worker - Starting")
    print(f"{'='*80}")
    print(f"Workspace: {workspace}")
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"{'='*80}\n")

    # Get idea UID from command line or environment variable
    idea_uid = None
    if len(sys.argv) > 1:
        idea_uid = sys.argv[1]
    elif os.getenv("IDEA_UID"):
        idea_uid = os.getenv("IDEA_UID")
    
    idea_name = os.getenv("IDEA_NAME", "unknown")

    # Verify required environment variables
    required_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "ANTHROPIC_API_KEY",
    ]

    # S3 is required for cloud mode
    required_vars.extend(["S3_BUCKET", "S3_ENDPOINT_URL"])

    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        print(f"❌ ERROR: Missing required environment variables:")
        for v in missing:
            print(f"   - {v}")
        sys.exit(1)

    from aar.config import DATASET_NAME, WEAK_MODEL, STRONG_MODEL, FULL_AUTO_MAX_RUNTIME_SECONDS

    print(f"✓ All required environment variables present")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Weak model: {WEAK_MODEL}")
    print(f"  Strong model: {STRONG_MODEL}")
    max_runtime = FULL_AUTO_MAX_RUNTIME_SECONDS
    print(f"  Max runtime: {max_runtime}s ({max_runtime / 3600:.1f} hours)")

    if not idea_uid:
        print(f"\n❌ ERROR: idea_uid is required")
        print(f"   Pass idea_uid as argument or set IDEA_UID env var")
        sys.exit(1)

    print(f"\n🎯 TARGET IDEA MODE")
    print(f"  Idea UID: {idea_uid}")
    print(f"  Idea Name: {idea_name}")
    print(f"  Claude will iterate on this specific idea for up to {max_runtime / 3600:.1f} hours")

    # Download prerequisites
    if not download_prerequisites():
        print(f"\n❌ Failed to download prerequisites")
        sys.exit(1)

    # Download target idea
    idea_data = download_idea(idea_uid)
    if not idea_data:
        print(f"\n❌ Failed to download target idea")
        sys.exit(1)
    # Update idea_name from downloaded data
    idea_name = idea_data.get("Name", idea_name)

    # Launch autonomous loop
    success = launch_autonomous_loop(idea_uid=idea_uid, idea_name=idea_name)

    # Upload final results and logs to S3
    upload_final_artifacts(idea_uid, idea_name, success)

    print(f"\n{'='*80}")
    print(f"Autonomous Research Worker - Finished")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().isoformat()}")
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*80}\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
