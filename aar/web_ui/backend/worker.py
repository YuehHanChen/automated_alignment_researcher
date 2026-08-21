"""Background worker for running experiments."""
import json
import os
import subprocess
import threading
import time
import traceback
import requests
from datetime import datetime, timezone
from pathlib import Path

from aar.infrastructure.runpod import RunPodCapacityError, RunPodPermanentError
from aar.web_ui.backend.models import db, Experiment, _safe_datetime_subtract
from aar.web_ui.backend import config


class ExperimentWorker:
    """Background worker that processes experiment queue."""

    def __init__(self, app):
        self.app = app
        self.running = False
        self.thread = None

    def start(self):
        """Start the worker thread."""
        if self.running:
            return

        self.running = True
        
        # Resume monitoring for any running experiments (e.g., after server restart)
        with self.app.app_context():
            # Only resume monitoring for actual RunPod pods (skip local- prefixed)
            running_experiments = Experiment.query.filter_by(status='running').filter(
                Experiment.pod_id.isnot(None)
            ).all()
            runpod_experiments = [e for e in running_experiments if e.pod_id and not e.pod_id.startswith('local-')]

            if runpod_experiments:
                print(f"🔄 Found {len(runpod_experiments)} running RunPod experiment(s). Resuming monitoring...")
                for i, experiment in enumerate(runpod_experiments):
                    if i > 0:
                        time.sleep(2)
                    print(f"   - Resuming monitoring for {experiment.idea_name} (pod: {experiment.pod_id})")
                    self._monitor_runpod_pod(experiment)
            else:
                print("✓ No running experiments to resume monitoring")
        
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        print("✓ Experiment worker started")

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("✓ Experiment worker stopped")

    def _worker_loop(self):
        """Main worker loop that processes queued experiments."""
        print("🔄 Worker loop started, polling for queued experiments...")

        while self.running:
            try:
                with self.app.app_context():
                    # Get next queued experiment
                    experiment = Experiment.query.filter_by(status='queued').order_by(
                        Experiment.queue_time
                    ).first()

                    if experiment:
                        # Check if this experiment was recently retried (capacity issue)
                        # Skip if retry delay hasn't elapsed yet
                        if experiment.last_deploy_attempt:
                            elapsed = _safe_datetime_subtract(datetime.now(), experiment.last_deploy_attempt)
                            if elapsed < config.POD_DEPLOY_RETRY_DELAY_SECONDS:
                                remaining = config.POD_DEPLOY_RETRY_DELAY_SECONDS - elapsed
                                # Don't print every poll cycle - just skip quietly
                                time.sleep(config.POLL_INTERVAL)
                                continue

                        print(f"📥 Found queued experiment: {experiment.idea_name} (id={experiment.id})")
                        if experiment.deploy_retry_count > 0:
                            print(f"   (Retry attempt {experiment.deploy_retry_count + 1}/{config.POD_DEPLOY_MAX_RETRIES})")
                        self._run_experiment(experiment)
                        # Sleep after processing (or skipping) an experiment to avoid rapid polling
                        # This interval controls how often we check the queue
                        time.sleep(config.POLL_INTERVAL)
                    else:
                        # No experiments queued, sleep
                        time.sleep(config.POLL_INTERVAL)
            except Exception as e:
                print(f"❌ Worker loop error: {e}")
                traceback.print_exc()
                time.sleep(5)  # Sleep longer on error before retrying

    def _run_experiment(self, experiment):
        """Run a single experiment."""
        print(f"\n{'=' * 80}")
        print(f"Starting experiment: {experiment.idea_name}")
        print(f"{'=' * 80}\n")

        # Check required credentials from environment (raise error if missing)
        # ANTHROPIC_API_KEY is required for orchestrator
        if not os.environ.get('ANTHROPIC_API_KEY'):
            raise ValueError("ANTHROPIC_API_KEY environment variable is required. Set it before starting the Flask server.")

        if config.DEPLOY_TO_RUNPOD:
            # AWS credentials are required for S3 operations (RunPod mode only)
            if not os.environ.get('AWS_ACCESS_KEY_ID'):
                raise ValueError("AWS_ACCESS_KEY_ID environment variable is required. Set it before starting the Flask server.")
            if not os.environ.get('AWS_SECRET_ACCESS_KEY'):
                raise ValueError("AWS_SECRET_ACCESS_KEY environment variable is required. Set it before starting the Flask server.")

            # WANDB_API_KEY is required for wandb logging (training will fail without it)
            if not os.environ.get('WANDB_API_KEY'):
                raise ValueError("WANDB_API_KEY environment variable is required. Set it before starting the Flask server.")

            # RUNPOD_API_KEY is required only if deploying to RunPod
            if not os.environ.get('RUNPOD_API_KEY'):
                raise ValueError("RUNPOD_API_KEY environment variable is required when DEPLOY_TO_RUNPOD=true. Set it before starting the Flask server.")

        # Check if experiment was killed before we start
        # (reload from DB in case it was modified)
        db.session.refresh(experiment)
        if experiment.status == 'failed':
            print(f"  Experiment {experiment.idea_name} was killed before starting. Skipping.")
            return False  # Not skipped due to limit, but killed

        # Check concurrent jobs limit BEFORE setting status to 'running'
        # This prevents the current experiment from being counted in the limit check
        running_count = Experiment.query.filter_by(status='running').count()
        max_concurrent = config.MAX_CONCURRENT_PODS
        if running_count >= max_concurrent:
            # Keep it queued - don't set to running
            return True  # Return True to indicate it was skipped

        # Update status to running (only after passing the concurrent pods check)
        experiment.status = 'running'
        experiment.start_time = datetime.now(timezone.utc)
        experiment.end_time = None  # Clear any previous end_time
        db.session.commit()

        logs = []

        try:
            # Load the idea details from DB first, then fall back to file
            idea_data = None
            
            # 1. Try to load from DB (idea_json field)
            if experiment.idea_json:
                try:
                    idea_data = json.loads(experiment.idea_json)
                    logs.append("Loaded idea from database")
                except json.JSONDecodeError:
                    pass
            
            # 2. Fall back to idea.json file
            if not idea_data:
                idea_dir = config.RESULTS_DIR / experiment.idea_name
                idea_json_path = idea_dir / "idea.json"

                if idea_json_path.exists():
                    with open(idea_json_path, 'r') as f:
                        idea_data = json.load(f)
                    logs.append(f"Loaded idea from file: {idea_json_path}")
            
            # 3. If still not found, create minimal idea data from experiment
            if not idea_data:
                idea_data = {
                    "Name": experiment.idea_name,
                    "Description": experiment.idea_description or "",
                }
                logs.append("Created minimal idea data from experiment record")

            logs.append(f"Loaded idea: {experiment.idea_name}")
            logs.append(f"Name: {idea_data.get('Name', 'N/A')}")

            # Use experiment's stored config (from user selection), not global config
            exp_dataset = experiment.dataset or config.DATASET_NAME
            exp_weak_model = experiment.weak_model or config.WEAK_MODEL
            exp_strong_model = experiment.strong_model or config.STRONG_MODEL
            exp_dataset_dir = f"{config.DATA_BASE_DIR}/{exp_dataset}"

            # Data directory path
            logs.append(f"Data Dir: {exp_dataset_dir}")
            logs.append(f"Weak Model: {exp_weak_model}")
            logs.append(f"Strong Model: {exp_strong_model}")
            logs.append(f"Seeds: {config.SEEDS}")
            logs.append(f"Epochs: {config.EPOCHS}")

            # Determine execution mode: per-experiment field overrides global config
            exec_mode = experiment.execution_mode
            if not exec_mode:
                # Fall back to global config
                if config.DEPLOY_TO_RUNPOD:
                    exec_mode = 'runpod'
                elif config.DOCKER_LOCAL_MODE:
                    exec_mode = 'docker'
                else:
                    exec_mode = 'local'

            MODE_LABELS = {
                'runpod': 'RunPod (cloud)',
                'docker': 'Docker (local GPUs)',
                'local': 'Local subprocess',
            }
            gpu_ids = experiment.gpu_ids  # e.g. "0,1,2,3" or None (all GPUs)
            logs.append(f"\nExecution mode: {MODE_LABELS.get(exec_mode, exec_mode)}")
            if gpu_ids:
                logs.append(f"GPUs: {gpu_ids}")

            if exec_mode == 'runpod':
                logs.append(f"\n{'=' * 80}")
                logs.append("Deploying autonomous worker to RunPod")
                logs.append(f"{'=' * 80}\n")
                experiment.logs = "\n".join(logs)
                db.session.commit()

                self._deploy_autonomous_worker_to_runpod(experiment, idea_data, logs)
            else:
                logs.append(f"\n{'=' * 80}")
                logs.append(f"Launching autonomous worker ({MODE_LABELS.get(exec_mode, exec_mode)})")
                logs.append(f"{'=' * 80}\n")
                experiment.logs = "\n".join(logs)
                db.session.commit()

                self._run_local_worker(experiment, idea_data, logs,
                                      use_docker=(exec_mode == 'docker'),
                                      gpu_ids=gpu_ids)

            # Refresh experiment to get pod_id if deployment succeeded
            db.session.refresh(experiment)

            # If deployment succeeded (pod_id is set), return early
            # The monitoring thread will handle completion
            if experiment.pod_id:
                print(f"\n{'=' * 80}")
                print(f"✓ Deployed experiment {experiment.idea_name} to RunPod pod {experiment.pod_id}")
                print(f"🔍 Starting monitoring for pod {experiment.pod_id}")
                print(f"{'=' * 80}\n")
                return False  # Successfully started, not skipped
            # If deployment failed, continue to finally block to mark as failed

            # Update logs
            experiment.logs = "\n".join(logs)
            db.session.commit()

        except Exception as e:
            error_trace = traceback.format_exc()
            logs.append(f"\n❌ Error: {str(e)}")
            logs.append(f"\nTraceback:\n{error_trace}")

            experiment.status = 'failed'
            experiment.error_msg = str(e)

            print(f"\n❌ Experiment failed: {experiment.idea_name}")
            print(error_trace)

        finally:
            # Refresh experiment to get latest state (especially pod_id for RunPod deployments)
            db.session.refresh(experiment)

            # If deployed to RunPod, don't finalize here - monitoring thread will handle it
            if experiment.pod_id:
                experiment.logs = "\n".join(logs)
                db.session.commit()
                return False  # Successfully started, not skipped

            # Deployment failed - update final state
            experiment.end_time = datetime.now(timezone.utc)
            experiment.logs = "\n".join(logs)
            db.session.commit()

            print(f"\n{'=' * 80}")
            print(f"Experiment finished: {experiment.idea_name}")
            print(f"Status: {experiment.status}")
            print(f"{'=' * 80}\n")

    def _monitor_runpod_pod(self, experiment):
        """
        Monitor RunPod pod status and retrieve results when complete.
        Runs in a separate thread to avoid blocking the worker loop.
        
        Args:
            experiment: Experiment database record with pod_id set
        """
        # Store experiment ID and pod_id before creating background thread
        # (SQLAlchemy objects can't be shared across threads)
        experiment_id = experiment.id
        pod_id = experiment.pod_id
        
        def monitor_thread():
            """Background thread to monitor pod status."""
            import time
            from aar.infrastructure.runpod import get_pod_status
            from aar.infrastructure.s3_utils import download_results
            
            def log_and_print(message):
                """Print message and append to experiment logs."""
                print(message)
                try:
                    with self.app.app_context():
                        experiment = db.session.get(Experiment, experiment_id)
                        if experiment:
                            current_logs = experiment.logs or ""
                            experiment.logs = current_logs + "\n" + message if current_logs else message
                            db.session.commit()
                except Exception as e:
                    # If logging fails, at least print to console
                    print(f"⚠️  Failed to append to experiment logs: {e}")
            
            def get_uid_from_experiment(experiment):
                """
                Get UID for an experiment. Checks in order:
                1. Database (fastest, most reliable)
                2. Local idea.json file
                3. S3 search (slowest, for backward compatibility)
                
                Returns:
                    UID string if found, None if not found
                """
                # First try: get from database (fastest, most reliable)
                if experiment.idea_uid:
                    return experiment.idea_uid
                
                # Second try: check local idea.json file
                try:
                    idea_dir = config.RESULTS_DIR / experiment.idea_name
                    idea_json_path = idea_dir / "idea.json"
                    if idea_json_path.exists():
                        with open(idea_json_path, 'r') as f:
                            idea_data = json.load(f)
                        if idea_data.get("uid"):
                            uid = idea_data["uid"]
                            # Store in database for next time
                            experiment.idea_uid = uid
                            db.session.commit()
                            log_and_print(f"   ✓ Found UID in local idea.json, stored in database: {uid}")
                            return uid
                except Exception as e:
                    log_and_print(f"   ⚠️  Error reading local idea.json: {e}")
                
                # Third try: search S3 (for experiments created before UID was stored in DB)
                log_and_print(f"   ⚠️  UID not in database or local file, searching S3 (this is slow, should only happen for old experiments)...")
                try:
                    from aar.infrastructure.s3_utils import get_s3_client
                    s3_client = get_s3_client()
                    prefix = config.S3_IDEAS_PREFIX
                    
                    # List objects in ideas/ prefix to find the one matching our idea_name
                    paginator = s3_client.get_paginator('list_objects_v2')
                    pages = paginator.paginate(Bucket=config.S3_BUCKET, Prefix=prefix, Delimiter='/')
                    
                    for page in pages:
                        # Check common prefixes (UIDs)
                        if 'CommonPrefixes' in page:
                            for common_prefix in page['CommonPrefixes']:
                                uid_candidate = common_prefix['Prefix'].replace(prefix, '').rstrip('/')
                                # Try to download idea.json for this UID
                                try:
                                    idea_key = f"{prefix}{uid_candidate}/idea.json"
                                    response = s3_client.get_object(Bucket=config.S3_BUCKET, Key=idea_key)
                                    idea_data = json.loads(response['Body'].read().decode('utf-8'))
                                    # Check if this idea matches our experiment
                                    if idea_data.get('Name') == experiment.idea_name:
                                        # Store in database for next time
                                        experiment.idea_uid = uid_candidate
                                        db.session.commit()
                                        
                                        # Also sync to local file to ensure consistency
                                        try:
                                            idea_dir = config.RESULTS_DIR / experiment.idea_name
                                            idea_dir.mkdir(parents=True, exist_ok=True)
                                            idea_json_path = idea_dir / "idea.json"
                                            with open(idea_json_path, 'w') as f:
                                                json.dump(idea_data, f, indent=2)
                                            log_and_print(f"   ✓ Synced UID to local file: {idea_json_path}")
                                        except Exception as e:
                                            log_and_print(f"   ⚠️  Could not sync to local file: {e}")
                                        
                                        log_and_print(f"   ✓ Found UID in S3, stored in database: {uid_candidate}")
                                        return uid_candidate
                                except Exception:
                                    continue  # Try next UID
                    
                    return None  # Not found
                except Exception as e:
                    log_and_print(f"   ⚠️  Error searching S3 for UID: {e}")
                    return None
            
            # Use captured variables from outer scope
            # (experiment_id and pod_id are captured, not experiment object)
            
            if not pod_id:
                return
            
            log_and_print(f"🔍 Starting monitoring for pod {pod_id}")
            log_and_print(f"💡 Tip: You can check the worker pod console logs in RunPod dashboard")
            log_and_print(f"   Dashboard: https://console.runpod.io/pods?id={pod_id}")

            max_wait_time = config.FULL_AUTO_POD_TIMEOUT_SECONDS
            check_interval = 60  # Check every minute

            # Get experiment start time from database (persists across server restarts)
            # This ensures timeout is based on actual experiment start, not monitoring resume time
            with self.app.app_context():
                experiment = db.session.get(Experiment, experiment_id)
                if experiment and experiment.start_time:
                    # Use experiment's actual start time (persists across restarts)
                    # Handle both timezone-aware and naive datetimes
                    start_dt = experiment.start_time
                    if start_dt.tzinfo is None:
                        # Naive datetime - assume UTC and make it aware
                        start_dt = start_dt.replace(tzinfo=timezone.utc)
                    experiment_start_timestamp = start_dt.timestamp()
                    log_and_print(f"   Experiment started at: {experiment.start_time.isoformat()}")
                else:
                    # Fallback: use current time if start_time not set (shouldn't happen)
                    experiment_start_timestamp = time.time()
                    log_and_print(f"   ⚠️  No start_time in database, using current time as fallback")
            
            while True:
                try:
                    with self.app.app_context():
                        # Reload experiment from database (can't use object from different thread/session)
                        experiment = db.session.get(Experiment, experiment_id)
                        if not experiment:
                            log_and_print(f"⚠️  Experiment {experiment_id} not found in database. Stopping monitoring.")
                            return
                        
                        # Check if experiment was killed
                        if experiment.status == 'failed' and 'killed' in (experiment.error_msg or '').lower():
                            log_and_print(f"🛑 Experiment {experiment.idea_name} was killed. Stopping monitoring.")
                            return
                        
                        # Get pod status
                        # Get pod status - if 404, pod was terminated/deleted
                        # Calculate elapsed time based on experiment's actual start time (not monitoring start)
                        current_time = time.time()
                        elapsed = int(current_time - experiment_start_timestamp)
                        elapsed_hours = elapsed / 3600
                        log_and_print(f"\n[{elapsed}s / {elapsed_hours:.1f}h] Checking pod {pod_id} status...")
                        
                        # Check if we've exceeded the timeout (based on experiment start time)
                        if elapsed >= max_wait_time:
                            log_and_print(f"⏱️  Timeout reached ({max_wait_time/3600:.1f}h elapsed). Marking as failed...")
                            break  # Exit loop to handle timeout
                        pod_state = None  # Initialize to None to detect if we didn't get a state
                        try:
                            pod_status = get_pod_status(
                                pod_id=pod_id,
                                runpod_api_key=os.environ.get('RUNPOD_API_KEY'),
                            )
                            pod_state = pod_status.get('desiredStatus', 'UNKNOWN')
                            log_and_print(f"   Pod state: {pod_state}")
                        except requests.exceptions.HTTPError as e:
                            # Pod not found (404) - pod was terminated/deleted (possibly manually killed)
                            status_code = e.response.status_code if e.response else None
                            error_str = str(e)
                            
                            # Check for 404 in multiple ways (status code or error message)
                            # This handles cases where response might be None or status_code check fails
                            is_404 = (status_code == 404) or ('404' in error_str) or ('Not Found' in error_str)
                            
                            if is_404:
                                log_and_print(f"⚠️  Pod {pod_id} not found (404) - pod was terminated/deleted")
                                log_and_print(f"   Someone may have manually killed the pod. Checking for results in S3...")
                                pod_state = 'TERMINATED'  # When pod is terminated, it's deleted
                                # Fall through to check for results, then mark as failed/cancelled if no results
                            else:
                                # Other HTTP error - log and continue monitoring
                                log_and_print(f"⚠️  Error checking pod {pod_id} status: {e} (status: {status_code})")
                                time.sleep(check_interval)
                                continue
                        except Exception as e:
                            # Check if it's a 404 error in the exception message (fallback)
                            error_str = str(e)
                            if '404' in error_str or 'Not Found' in error_str:
                                log_and_print(f"⚠️  Pod {pod_id} not found (404) - pod was terminated/deleted")
                                log_and_print(f"   Someone may have manually killed the pod. Checking for results in S3...")
                                pod_state = 'TERMINATED'
                            else:
                                # Other error - log and continue monitoring
                                log_and_print(f"⚠️  Error checking pod {pod_id} status: {e}")
                                time.sleep(check_interval)
                                continue
                        
                        # If we still don't have a pod_state, something went wrong - skip this iteration
                        if pod_state is None:
                            log_and_print(f"⚠️  Could not determine pod state for {pod_id}. Skipping this check.")
                            time.sleep(check_interval)
                            continue

                        # Check S3 for results.json to detect completion, then terminate pod.
                        results_found_in_s3 = False
                        if pod_state in ['RUNNING', 'RUNNING_IN_QUEUE']:
                            try:
                                from aar.infrastructure.s3_utils import get_s3_client
                                s3_client = get_s3_client()
                                uid = experiment.idea_uid
                                run_id = experiment.run_id
                                if uid and run_id:
                                    results_key = f"{config.S3_IDEAS_PREFIX}{uid}/{run_id}/results.json"
                                    log_and_print(f"   Checking S3 for results at: s3://{config.S3_BUCKET}/{results_key}")
                                    s3_client.head_object(Bucket=config.S3_BUCKET, Key=results_key)
                                    # Results exist! Worker has completed, proceed to sync and terminate
                                    log_and_print(f"✓ Found results.json in S3 - worker has completed!")
                                    results_found_in_s3 = True
                            except Exception as e:
                                error_str = str(e)
                                if '404' in error_str or 'NoSuchKey' in error_str or 'Not Found' in error_str:
                                    # Results not yet uploaded, continue waiting
                                    log_and_print(f"🤖 Worker pod is running, no results.json yet. Next check in {check_interval}s...")
                                else:
                                    log_and_print(f"⚠️  Error checking S3 for results: {e}")
                                time.sleep(check_interval)
                                continue

                        if pod_state in ['STOPPED', 'TERMINATED', 'COMPLETED'] or results_found_in_s3:
                            log_and_print(f"✓ Worker pod state: {pod_state}, results_in_s3: {results_found_in_s3}")

                            # If results found in S3, mark as completed (worker finished successfully)
                            # Otherwise, mark based on pod state
                            if results_found_in_s3:
                                experiment.status = 'completed'
                                experiment.error_msg = None
                                finish_reason = "results.json found in S3"
                            elif pod_state == 'TERMINATED':
                                experiment.status = 'failed'
                                experiment.error_msg = 'Worker pod was terminated/deleted before completion.'
                                finish_reason = f"pod {pod_state}"
                            else:
                                experiment.status = 'completed'
                                experiment.error_msg = None
                                finish_reason = f"pod {pod_state}"

                            experiment.end_time = datetime.now(timezone.utc)
                            current_logs = experiment.logs or ""
                            experiment.logs = (
                                current_logs
                                + "\n\n"
                                + ("=" * 80)
                                + f"\nWorker finished ({finish_reason})\n"
                                + ("=" * 80)
                                + "\n"
                            )
                            db.session.commit()

                            # Sync results from S3 results.json to Experiment record
                            try:
                                log_and_print("📥 Syncing results from S3 results.json...")
                                uid = experiment.idea_uid
                                run_id = experiment.run_id
                                if uid and run_id:
                                    s3_key = f"{config.S3_IDEAS_PREFIX}{uid}/{run_id}/results.json"
                                    from aar.infrastructure.s3_utils import download_file_from_s3
                                    import tempfile
                                    import json as json_module

                                    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                                        tmp_path = tmp.name

                                    download_file_from_s3(
                                        s3_key=s3_key,
                                        local_path=tmp_path,
                                        bucket_name=config.S3_BUCKET,
                                    )

                                    with open(tmp_path) as f:
                                        results = json_module.load(f)

                                    # Extract metrics from results.json
                                    best_pgr = results.get("pgr")
                                    total_iterations = results.get("total_iterations", 0)
                                    best_iter_num = results.get("best_iteration", 0)

                                    if best_pgr is not None:
                                        experiment.pgr = best_pgr
                                        experiment.status = 'completed'
                                        log_and_print(f"   ✓ Best PGR: {best_pgr:.4f} (iteration {best_iter_num})")

                                    if results.get("transfer_acc") is not None:
                                        experiment.transfer_acc = results["transfer_acc"]
                                    if results.get("weak_acc") is not None:
                                        experiment.weak_acc = results["weak_acc"]
                                    if results.get("strong_acc") is not None:
                                        experiment.strong_acc = results["strong_acc"]

                                    experiment.results_uploaded_to_s3 = True
                                    experiment.logs = (experiment.logs or "") + f"\n✓ Synced from results.json: PGR={best_pgr}, iterations={total_iterations}\n"
                                    db.session.commit()
                                    log_and_print(f"   ✓ Results synced to WebUI database")

                                    # Clean up temp file
                                    import os as os_module
                                    os_module.unlink(tmp_path)
                                else:
                                    log_and_print("   ⚠️  Missing uid or run_id, cannot sync results")
                            except Exception as e:
                                log_and_print(f"   ⚠️  Failed to sync results from S3: {e}")
                                # Don't fail the whole process, just log the error

                            # Delete the pod to avoid lingering costs
                            try:
                                from aar.infrastructure.runpod import delete_pod
                                log_and_print(f"🗑️  Deleting worker pod {pod_id}...")
                                delete_pod(pod_id=pod_id, runpod_api_key=os.environ.get('RUNPOD_API_KEY'))
                                log_and_print(f"✓ Pod {pod_id} deleted successfully")
                            except Exception as e:
                                log_and_print(f"⚠️  Failed to delete pod {pod_id}: {e}")

                            return  # Done monitoring

                        # Unknown state, wait and check again
                        log_and_print(f"   Pod {pod_id} in unknown/unhandled state: {pod_state}")
                        log_and_print(f"   Will check again in {check_interval}s...")
                        time.sleep(check_interval)
                        
                except Exception as e:
                    log_and_print(f"⚠️  Error monitoring pod {pod_id}: {e}")
                    time.sleep(check_interval)
            
            # Timeout - mark as failed and try to delete pod
            with self.app.app_context():
                # Reload experiment from database (can't use object from different thread/session)
                experiment = Experiment.query.get(experiment_id)
                if not experiment:
                    log_and_print(f"⚠️  Experiment {experiment_id} not found in database. Cannot mark as timed out.")
                    return
                
                experiment.status = 'failed'
                experiment.end_time = datetime.now(timezone.utc)
                
                # Calculate actual elapsed time
                if experiment.start_time:
                    elapsed_seconds = _safe_datetime_subtract(experiment.end_time, experiment.start_time)
                    elapsed_hours = elapsed_seconds / 3600
                    elapsed_str = f"{elapsed_hours:.1f} hours ({elapsed_seconds:.0f}s)"
                else:
                    elapsed_str = "unknown"
                
                timeout_hours = max_wait_time / 3600
                experiment.error_msg = f'Pod monitoring timeout (configured: {timeout_hours:.1f}h, actual elapsed: {elapsed_str})'

                # Append timeout message to logs
                current_logs = experiment.logs or ""
                timeout_msg = f"\n\n{'='*80}\n⏱️  Monitoring timeout after {elapsed_str} (configured limit: {timeout_hours:.1f}h)\n{'='*80}"
                experiment.logs = current_logs + timeout_msg if current_logs else timeout_msg
                db.session.commit()
                
                # Try to delete the pod if it's still running
                if pod_id:
                    try:
                        from aar.infrastructure.runpod import delete_pod
                        log_and_print(f"🗑️  Attempting to delete timed-out pod {pod_id}...")
                        delete_pod(
                            pod_id=pod_id,
                            runpod_api_key=os.environ.get('RUNPOD_API_KEY'),
                        )
                        log_and_print(f"✓ Successfully deleted timed-out pod {pod_id}")
                        experiment.error_msg += " (pod deleted)"
                    except Exception as e:
                        log_and_print(f"⚠️  Failed to delete timed-out pod {pod_id}: {e}")
                        experiment.error_msg += f" (pod deletion failed: {str(e)})"
                
                db.session.commit()
                log_and_print(f"⏱️  Monitoring timeout for {experiment.idea_name}")
        
        # Start monitoring thread
        monitor = threading.Thread(target=monitor_thread, daemon=True)
        monitor.start()
        print(f"✓ Started monitoring thread for pod {experiment.pod_id}")

    def _run_local_worker(self, experiment, idea_data, logs, use_docker=False, gpu_ids=None):
        """
        Run autonomous agent loop locally using local GPUs.

        Args:
            use_docker: If True, run inside a Docker container with GPU access.
                        Only data/ (no labels) is mounted. If False, run as bare subprocess.
            gpu_ids: Comma-separated GPU IDs (e.g. "0,1,2,3"). None means all GPUs.

        The agent connects back to this server for AAR evaluation.
        """
        import uuid
        import sys

        # Ensure idea has a UID
        uid = experiment.idea_uid
        if not uid:
            uid = idea_data.get("uid") or str(uuid.uuid4())[:8]
            idea_data["uid"] = uid
            experiment.idea_uid = uid
            db.session.commit()

        run_id = f"local-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        experiment.run_id = run_id

        # Write idea.json to the target_idea directory so the agent can find it
        from aar.config import TARGET_IDEA_FILE
        idea_dir = Path(TARGET_IDEA_FILE).parent
        idea_dir.mkdir(parents=True, exist_ok=True)
        idea_file = idea_dir / "idea.json"
        with open(idea_file, 'w') as f:
            json.dump(idea_data, f, indent=2)
        logs.append(f"Wrote idea to: {idea_file}")

        exp_dataset = experiment.dataset or config.DATASET_NAME
        exp_weak_model = experiment.weak_model or config.WEAK_MODEL
        exp_strong_model = experiment.strong_model or config.STRONG_MODEL

        # Common env vars for both modes
        worker_env = {
            "DATASET_NAME": exp_dataset,
            "WEAK_MODEL": exp_weak_model,
            "STRONG_MODEL": exp_strong_model,
            "LOCAL_MODE": "true",
            "RUN_ID": run_id,
            "IDEA_UID": uid,
            "IDEA_NAME": experiment.idea_name,
        }

        # Pass through API keys from server env
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "WANDB_API_KEY"]:
            if os.environ.get(key):
                worker_env[key] = os.environ[key]

        if use_docker:
            cmd, container_name = self._build_docker_cmd(
                uid, run_id, experiment.idea_name, exp_dataset, worker_env, logs,
                gpu_ids=gpu_ids,
            )
        else:
            # Bare subprocess mode
            worker_env["DATA_DIR"] = f"{config.DATA_BASE_DIR}/{exp_dataset}"
            worker_env["ORCHESTRATOR_API_URL"] = f"http://localhost:{os.environ.get('PORT', '8000')}"
            env = os.environ.copy()
            env.update(worker_env)
            # Remove CLAUDECODE env var to allow nested Claude Code launches
            env.pop("CLAUDECODE", None)
            # Disable Python output buffering so logs stream in real-time
            env["PYTHONUNBUFFERED"] = "1"
            # Set GPU visibility
            if gpu_ids:
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids
            cmd = [
                sys.executable, "run.py", "agent",
                "--idea-uid", uid,
                "--idea-name", experiment.idea_name,
                "--local",
            ]
            container_name = None

        logs.append(f"Mode: {'Docker' if use_docker else 'subprocess'}")
        logs.append(f"Command: {' '.join(cmd)}")
        logs.append(f"Dataset: {exp_dataset}")
        logs.append(f"Weak model: {exp_weak_model}")
        logs.append(f"Strong model: {exp_strong_model}")
        experiment.logs = "\n".join(logs)
        db.session.commit()

        # Store values for the background thread (no SQLAlchemy objects)
        experiment_id = experiment.id
        experiment_name = experiment.idea_name
        workspace_dir = config.WORKSPACE_DIR
        # For subprocess mode, capture full env
        proc_env = env if not use_docker else None

        def local_worker_thread():
            """Background thread running the local worker."""
            try:
                proc = subprocess.Popen(
                    cmd,
                    env=proc_env,
                    cwd=workspace_dir if not use_docker else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                # Stream output to experiment logs
                pending_lines = []
                last_flush = time.time()
                for line in proc.stdout:
                    line = line.rstrip()
                    pending_lines.append(line)
                    print(f"[local-worker] {line}")

                    # Flush to DB every 2 seconds or every 10 lines
                    if time.time() - last_flush >= 2 or len(pending_lines) >= 10:
                        try:
                            with self.app.app_context():
                                exp = db.session.get(Experiment, experiment_id)
                                if exp:
                                    current = exp.logs or ""
                                    exp.logs = current + "\n" + "\n".join(pending_lines)
                                    db.session.commit()
                            pending_lines = []
                            last_flush = time.time()
                        except Exception:
                            pass

                proc.wait()

                with self.app.app_context():
                    exp = db.session.get(Experiment, experiment_id)
                    if exp:
                        # Flush any remaining lines
                        if pending_lines:
                            current = exp.logs or ""
                            exp.logs = current + "\n" + "\n".join(pending_lines)

                        if proc.returncode == 0:
                            exp.status = 'completed'
                        else:
                            exp.status = 'failed'
                            exp.error_msg = f"Local worker exited with code {proc.returncode}"

                        exp.end_time = datetime.now(timezone.utc)
                        db.session.commit()

                print(f"Local worker finished for {experiment_name} (exit={proc.returncode})")

            except Exception as e:
                print(f"Local worker error: {e}")
                traceback.print_exc()
                try:
                    with self.app.app_context():
                        exp = db.session.get(Experiment, experiment_id)
                        if exp:
                            exp.status = 'failed'
                            exp.error_msg = str(e)
                            exp.end_time = datetime.now(timezone.utc)
                            db.session.commit()
                except Exception:
                    pass
            finally:
                # Cleanup Docker container if it was used
                if use_docker and container_name:
                    subprocess.Popen(
                        f"(timeout 60 {config.DOCKER_EXECUTABLE} stop {container_name} "
                        f"|| {config.DOCKER_EXECUTABLE} rm -f {container_name}) >/dev/null 2>&1 &",
                        shell=True,
                    )

        # Set a local pod_id so the finally block in _run_experiment doesn't
        # overwrite status (it skips finalization when pod_id is set)
        experiment.pod_id = f"local-{run_id}"
        db.session.commit()

        thread = threading.Thread(target=local_worker_thread, daemon=True)
        thread.start()
        print(f"Started local worker thread for {experiment_name}")

    def _build_docker_cmd(self, uid, run_id, idea_name, dataset, env_vars, logs, gpu_ids=None):
        """
        Build a `docker run` command for Docker local mode.

        Uses the same Dockerfile as RunPod. The entrypoint switches to ubuntu-cmd
        user and runs in /workspace/automated-w2s-research.

        Mounts:
        - data/{dataset}  -> /workspace/automated-w2s-research/data/{dataset}  (NO labels)
        - cache_results/  -> /workspace/automated-w2s-research/cache_results/  (baseline caches)
        - target_idea/    -> /workspace/automated-w2s-research/aar/research_loop/target_idea/
        - HF cache        -> shared to avoid re-downloading models

        Does NOT mount labeled_data/ — ground truth stays server-side only.
        """
        import uuid as uuid_mod

        container_name = f"w2s-local-{uid[:8]}-{uuid_mod.uuid4().hex[:6]}"
        workspace_dir = config.WORKSPACE_DIR
        docker = config.DOCKER_EXECUTABLE
        image = config.DOCKER_LOCAL_IMAGE
        # Code lives in /opt/automated-w2s-research inside the image (entrypoint copies to /workspace,
        # but volume mounts can interfere). Run directly from /opt/automated-w2s-research.
        container_ws = "/opt/automated-w2s-research"

        # The agent command to run inside the container
        agent_cmd = (
            f"cd {container_ws} && python run.py agent"
            f" --idea-uid {uid}"
            f" --idea-name '{idea_name}'"
            f" --local"
        )

        # Use host networking so the container can reach the server on localhost
        # Skip entrypoint to avoid /workspace copy conflicts
        # Run as ubuntu-cmd (Claude Code CLI won't run as root)
        cmd = [
            docker, "run",
            "--name", container_name,
            "--rm",
            "--entrypoint", "",
            "--user", "ubuntu-cmd",
            "--gpus", f'"device={gpu_ids}"' if gpu_ids else "all",
            "--network", "host",
            # Mount data (NO labels) read-only
            "-v", f"{workspace_dir}/data/{dataset}:{container_ws}/data/{dataset}:ro",
            # Mount cache_results read-only
            "-v", f"{workspace_dir}/cache_results:{container_ws}/cache_results:ro",
            # Mount target_idea so the agent can read idea.json
            "-v", f"{workspace_dir}/aar/research_loop/target_idea:"
                  f"{container_ws}/aar/research_loop/target_idea:ro",
            # Share HuggingFace model cache to avoid re-downloading
            "-v", f"{Path.home()}/.cache/huggingface:/home/ubuntu-cmd/.cache/huggingface",
            # Mount logs directory so server can read usage_stats.json
            "-v", f"{workspace_dir}/aar/research_loop/logs:"
                  f"{container_ws}/aar/research_loop/logs",
        ]

        # Ensure mounted dirs and files are writable by container user (uid may differ)
        for d in [
            Path(workspace_dir) / "aar" / "research_loop" / "logs",
            Path(workspace_dir) / "aar" / "research_loop" / "target_idea",
            Path(workspace_dir) / "cache_results",
        ]:
            d.mkdir(parents=True, exist_ok=True)
            subprocess.run(["chmod", "-R", "777", str(d)], timeout=10)

        # Set env vars for inside the container
        env_vars["DATA_DIR"] = f"{container_ws}/data/{dataset}"
        env_vars["WORKSPACE_DIR"] = container_ws
        env_vars["PYTHONPATH"] = container_ws
        env_vars["ORCHESTRATOR_API_URL"] = f"http://localhost:{os.environ.get('PORT', '8000')}"
        # Unset CLAUDECODE to allow nested Claude Code launches
        env_vars["CLAUDECODE"] = ""

        # Write env vars to a temp file and use --env-file (handles special chars in values)
        import tempfile
        env_file = tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False, prefix='w2s_docker_')
        for key, value in env_vars.items():
            env_file.write(f"{key}={value}\n")
        env_file.close()
        cmd.extend(["--env-file", env_file.name])

        cmd.extend([image, "bash", "-c", agent_cmd])

        logs.append(f"Docker container: {container_name}")
        logs.append(f"Docker image: {image}")
        logs.append(f"Mounts: data/{dataset} (ro), cache_results (ro), target_idea (ro), HF cache (rw)")
        logs.append(f"Network: host (agent reaches server on localhost)")

        return cmd, container_name

    def _deploy_autonomous_worker_to_runpod(self, experiment, idea_data, logs):
        """
        Deploy a long-running autonomous worker pod to RunPod.

        This runs `python -m aar.infrastructure.execute_autonomous` which
        launches the autonomous agent loop using Claude Agent SDK and keeps iterating
        for up to FULL_AUTO_WORKER_MAX_RUNTIME_SECONDS (default: 5 days).

        The selected idea is uploaded to S3 and passed to the worker, so Claude can
        continuously iterate and improve on that specific idea.
        """
        try:
            from aar.infrastructure.runpod import deploy_pod
            from aar.infrastructure.s3_utils import ensure_idea_has_uid, upload_idea_by_uid, idea_exists_in_s3

            logs.append(f"\n{'='*80}")
            logs.append("FULL AUTO WORKER MODE: Deploying autonomous worker pod")
            logs.append(f"Idea to improve: {experiment.idea_name}")
            logs.append(f"{'='*80}")

            # Generate a unique run_id for this execution
            run_id = f"auto-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            experiment.run_id = run_id
            logs.append(f"Generated RUN_ID: {run_id}")

            # Ensure idea has UID and upload to S3 (same as normal mode)
            uid = None
            if experiment.idea_uid:
                uid = experiment.idea_uid
                idea_data["uid"] = uid
                logs.append(f"✓ Using UID from database: {uid}")
            else:
                uid = ensure_idea_has_uid(idea_data)
                logs.append(f"✓ Generated new UID: {uid}")
            
            experiment.idea_uid = uid
            db.session.commit()
            
            # Upload idea to S3 if not already there
            if not idea_exists_in_s3(uid, config.S3_BUCKET, config.S3_IDEAS_PREFIX):
                logs.append(f"Uploading idea to S3...")
                upload_idea_by_uid(
                    idea=idea_data,
                    bucket_name=config.S3_BUCKET,
                    prefix=config.S3_IDEAS_PREFIX,
                    metadata={
                        "experiment_id": experiment.id,
                        "idea_name": experiment.idea_name,
                        "full_auto_mode": True,
                    },
                )
                logs.append(f"✓ Idea uploaded to S3: s3://{config.S3_BUCKET}/{config.S3_IDEAS_PREFIX}{uid}/idea.json")
            else:
                logs.append(f"✓ Idea already exists in S3")

            # Use experiment's stored config (from user selection), not global config
            exp_dataset = experiment.dataset or config.DATASET_NAME
            exp_weak_model = experiment.weak_model or config.WEAK_MODEL
            exp_strong_model = experiment.strong_model or config.STRONG_MODEL

            command = ["python", "-m", "aar.infrastructure.execute_autonomous", uid]
            logs.append(f"Command: {' '.join(command)}")

            env_vars = {
                # Core configuration (consumed by research_loop/config.py via env overrides)
                "DATASET_NAME": exp_dataset,
                "WEAK_MODEL": exp_weak_model,
                "STRONG_MODEL": exp_strong_model,
                "SEEDS": json.dumps(config.SEEDS),
                "EPOCHS": str(config.EPOCHS),
                "NUM_GPUS": str(config.NUM_GPUS),
                # AAR mode flag (autonomous baseline supports AAR mode)
                "AAR_MODE": "true" if config.AAR_MODE else "false",
                # Workspace paths (worker pod)
                "WORKSPACE_DIR": str(config.WORKSPACE_DIR),
                "DATA_DIR": str(Path(config.WORKSPACE_DIR) / "data" / exp_dataset),
                # Runtime control (enforced inside execute_autonomous)
                "FULL_AUTO_MAX_RUNTIME_SECONDS": str(config.FULL_AUTO_WORKER_MAX_RUNTIME_SECONDS),
                # Idea to improve (UID for downloading from S3)
                "IDEA_UID": uid,
                "IDEA_NAME": experiment.idea_name,
                # S3 config (used for downloads/uploads)
                "S3_BUCKET": config.S3_BUCKET,
                "S3_ENDPOINT_URL": config.S3_ENDPOINT_URL,
                "S3_IDEAS_PREFIX": config.S3_IDEAS_PREFIX,
                "AWS_REGION": config.S3_REGION,
                # Run ID for consistent tracking between web UI and agent loop
                "RUN_ID": run_id,
            }

            # Add ORCHESTRATOR_API_URL for remote evaluation (same as regular worker mode)
            orchestrator_api_url = os.environ.get('ORCHESTRATOR_API_URL') or config.ORCHESTRATOR_API_URL
            if orchestrator_api_url:
                env_vars["ORCHESTRATOR_API_URL"] = orchestrator_api_url
                logs.append(f"✓ ORCHESTRATOR_API_URL: Set ({orchestrator_api_url[:50]}...)")
            else:
                logs.append("⚠️  ORCHESTRATOR_API_URL not set - remote evaluation will not be available")

            # Required secrets for worker pod
            if 'ANTHROPIC_API_KEY' not in os.environ:
                error_msg = "ANTHROPIC_API_KEY environment variable is required for autonomous worker pod."
                logs.append(f"\n❌ ERROR: {error_msg}")
                experiment.logs = "\n".join(logs)
                db.session.commit()
                raise ValueError(error_msg)
            env_vars["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

            aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            if not aws_access_key or not aws_secret_key:
                error_msg = "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are required for autonomous worker pod."
                logs.append(f"\n❌ ERROR: {error_msg}")
                experiment.logs = "\n".join(logs)
                db.session.commit()
                raise ValueError(error_msg)
            env_vars["AWS_ACCESS_KEY_ID"] = aws_access_key
            env_vars["AWS_SECRET_ACCESS_KEY"] = aws_secret_key

            if 'WANDB_API_KEY' not in os.environ:
                error_msg = "WANDB_API_KEY environment variable is required for autonomous worker pod."
                logs.append(f"\n❌ ERROR: {error_msg}")
                experiment.logs = "\n".join(logs)
                db.session.commit()
                raise ValueError(error_msg)
            env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

            runpod_api_key = os.environ.get('RUNPOD_API_KEY')
            if not runpod_api_key:
                error_msg = "RUNPOD_API_KEY environment variable is required. Set it before starting the Flask server."
                logs.append(f"\n❌ ERROR: {error_msg}")
                experiment.logs = "\n".join(logs)
                db.session.commit()
                raise ValueError(error_msg)
            env_vars["RUNPOD_API_KEY"] = runpod_api_key

            # Deploy pod
            timeout_hours = int(getattr(config, "FULL_AUTO_POD_TIMEOUT_SECONDS", 5 * 24 * 3600) // 3600)
            pod_name = f"liangq-auto-{timeout_hours}h-{uid[:8]}-{run_id}"

            logs.append(f"\n{'='*80}")
            logs.append("STEP: Deploying RunPod pod (full auto worker)")
            logs.append(f"{'='*80}")
            logs.append(f"  Pod name: {pod_name}")
            logs.append(f"  Template: {config.RUNPOD_TEMPLATE_ID}")
            logs.append(f"  GPU: {config.RUNPOD_GPU_TYPE} x {config.NUM_GPUS}")
            logs.append(f"  Runtime limit: {config.FULL_AUTO_WORKER_MAX_RUNTIME_SECONDS}s")
            experiment.logs = "\n".join(logs)
            db.session.commit()

            pod_response = deploy_pod(
                command=command,
                env_vars=env_vars,
                pod_name=pod_name,
                template_id=config.RUNPOD_TEMPLATE_ID,
                gpu_count=config.NUM_GPUS,
                gpu_type_ids=[config.RUNPOD_GPU_TYPE],
                runpod_api_key=runpod_api_key,
            )

            pod_id = pod_response.get('id', 'unknown')
            experiment.pod_id = pod_id
            experiment.status = 'running'
            experiment.end_time = None
            experiment.logs = "\n".join(logs + [
                f"\n✓ Autonomous worker pod deployed",
                f"Pod ID: {pod_id}",
                f"RunPod Console: https://console.runpod.io/pods?id={pod_id}",
                f"Note: This mode does not produce a single results.json for the UI; it uploads artifacts continuously during the loop.",
            ])
            db.session.commit()

            print(f"✓ Deployed autonomous worker to RunPod pod {pod_id}")

            # Start monitoring the pod in background
            self._monitor_runpod_pod(experiment)

        except RunPodCapacityError as e:
            # Transient error - keep queued for retry
            experiment.deploy_retry_count = (experiment.deploy_retry_count or 0) + 1
            experiment.last_deploy_attempt = datetime.now()

            if experiment.deploy_retry_count >= config.POD_DEPLOY_MAX_RETRIES:
                logs.append(f"\n❌ MAX RETRIES EXCEEDED ({config.POD_DEPLOY_MAX_RETRIES})")
                logs.append(f"Error: {str(e)}")
                experiment.status = 'failed'
                experiment.error_msg = f"Capacity unavailable after {config.POD_DEPLOY_MAX_RETRIES} retries: {str(e)}"
            else:
                logs.append(f"\n⏳ CAPACITY UNAVAILABLE - WILL RETRY")
                logs.append(f"Attempt {experiment.deploy_retry_count}/{config.POD_DEPLOY_MAX_RETRIES}")
                logs.append(f"Next retry in {config.POD_DEPLOY_RETRY_DELAY_SECONDS}s")
                logs.append(f"Error: {str(e)}")
                experiment.status = 'queued'

            experiment.logs = "\n".join(logs)
            db.session.commit()

        except RunPodPermanentError as e:
            logs.append(f"\n❌ PERMANENT DEPLOYMENT ERROR")
            logs.append(f"Error: {str(e)}")
            experiment.status = 'failed'
            experiment.error_msg = str(e)
            experiment.logs = "\n".join(logs)
            db.session.commit()

        except Exception as e:
            error_trace = traceback.format_exc()
            logs.append(f"\n❌ DEPLOYMENT FAILED (UNKNOWN ERROR)")
            logs.append(f"Error: {str(e)}")
            logs.append(f"\nTraceback:\n{error_trace}")
            experiment.status = 'failed'
            experiment.error_msg = f"RunPod deployment failed: {str(e)}"
            experiment.logs = "\n".join(logs)
            db.session.commit()
