"""Orchestrator-side eval: launch an eval for a submitted model and wait for
its scores. Transport-aware:

- fs  -> submit a Slurm eval job (sbatch EVAL_SLURM_SCRIPT) that runs the
         entrypoint as/with access to the secret HOLDOUT_DIR, then poll the
         shared-FS scores.json.
- s3  -> spawn an ephemeral RunPod eval pod (runpod.deploy_pod), then poll S3.

The orchestrator never reads the secret suite or runs GPU work itself; it only
launches the eval and relays the aggregate scores.
"""
from __future__ import annotations

import subprocess
import time
from typing import Any, Optional

from aar import config, transport


def _spawn_fs(run_id: str, suite: str) -> str:
    """Submit a Slurm eval job. Returns a label (job id if parseable)."""
    cmd = ["sbatch", "--parsable",
           "--job-name", f"aareval-{run_id}",
           config.EVAL_SLURM_SCRIPT, run_id, suite]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return f"slurm:{out.stdout.strip()}"
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # No Slurm (e.g. local dev) — run the eval inline as a subprocess.
        subprocess.Popen(
            ["python", "-m", "aar.eval_pod.entrypoint", "--run-id", run_id, "--suite", suite]
        )
        return f"local:{getattr(e, 'returncode', 'noslurm')}"


def _spawn_s3(run_id: str, suite: str) -> str:
    from aar.infrastructure import runpod
    env = {
        "HARNESS_TRANSPORT": "s3",
        "S3_BUCKET": config.S3_BUCKET,
        "S3_ENDPOINT_URL": config.S3_ENDPOINT_URL,
        "AWS_ACCESS_KEY_ID": config.AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": config.AWS_SECRET_ACCESS_KEY,
        "SECRET_DATA_PREFIX": config.SECRET_DATA_PREFIX,
        "SUBMISSION_PREFIX": config.SUBMISSION_PREFIX,
        "SUITE_NAME": suite,
    }
    if config.OAI_API_KEY:
        env["OAI_API"] = config.OAI_API_KEY
        env["JUDGE_MODEL"] = config.JUDGE_MODEL
    resp = runpod.deploy_pod(
        command=["python", "-m", "aar.eval_pod.entrypoint", "--run-id", run_id, "--suite", suite],
        env_vars=env,
        pod_name=f"eval-{run_id}",
        template_id=config.EVAL_POD_TEMPLATE_ID,
        gpu_count=config.EVAL_POD_GPU_COUNT,
        gpu_type_ids=[config.EVAL_POD_GPU_TYPE],
    )
    return f"pod:{resp.get('id') or resp.get('podId') or ''}"


def spawn_eval(run_id: str, suite: str) -> str:
    return _spawn_fs(run_id, suite) if config.HARNESS_TRANSPORT == "fs" else _spawn_s3(run_id, suite)


def poll_scores(run_id: str, timeout: Optional[int] = None, interval: Optional[int] = None) -> dict[str, Any]:
    """Block until scores.json appears (via transport), or raise TimeoutError."""
    timeout = timeout or config.EVAL_POD_TIMEOUT_SECONDS
    interval = interval or config.EVAL_POLL_INTERVAL_SECONDS
    deadline = time.time() + timeout
    while time.time() < deadline:
        scores = transport.read_scores(run_id)
        if scores is not None:
            return scores
        time.sleep(interval)
    raise TimeoutError(f"eval scores for run {run_id!r} not found within {timeout}s")


def evaluate_model(run_id: str, suite: Optional[str] = None) -> dict[str, Any]:
    """Launch the eval and wait for the composite. Assumes the model is already
    published for run_id (transport.put_model done by the caller)."""
    suite = suite or config.SUITE_NAME
    if config.EVAL_VIA_WORKER and config.HARNESS_TRANSPORT == "fs":
        # Hardened two-user flow: a separate eval worker (running as the secret-
        # holding user) is already draining the queue and scoring submissions
        # against the mode-700 holdout. The research side cannot read the holdout,
        # so it must NOT spawn its own eval — it only polls for the worker's
        # scores. (put_model was already done by the caller, which is the signal
        # the worker picks up.)
        scores = poll_scores(run_id)
        scores["eval_launch"] = "worker"
        return scores
    launch = spawn_eval(run_id, suite)
    scores = poll_scores(run_id)
    scores["eval_launch"] = launch
    return scores
