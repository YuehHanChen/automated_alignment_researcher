"""
Configuration for the W2S research system.

Single source of truth — all modules import from here.
All values can be overridden via environment variables.
"""
import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

_REPO_ROOT = str(Path(__file__).parent.parent)
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", _REPO_ROOT)

# =============================================================================
# Dataset & Models
# =============================================================================

DATASET_NAME = os.getenv("DATASET_NAME", "chat")   # legacy W2S single-dataset (kept for back-compat)
DATA_DIR = os.getenv("DATA_DIR", f"{WORKSPACE_DIR}/data/{DATASET_NAME}")
WEAK_MODEL = os.getenv("WEAK_MODEL", "Qwen/Qwen1.5-0.5B-Chat")
STRONG_MODEL = os.getenv("STRONG_MODEL", "Qwen/Qwen3-4B-Base")
SEEDS = [42, 43, 44, 45, 46]

# Multi-benchmark suite (generalized harness). A suite YAML lists benchmarks of
# mixed categories that the AAR hillclimbs together; the target model is what
# methods modify.
SUITE_NAME = os.getenv("SUITE_NAME", "toy")
SUITE_CONFIG = os.getenv("SUITE_CONFIG", f"{WORKSPACE_DIR}/configs/{SUITE_NAME}.yaml")
TARGET_MODEL = os.getenv("TARGET_MODEL", WEAK_MODEL)

# =============================================================================
# Isolation boundary + eval pod
# =============================================================================
# The secret suite (test inputs + answers + rubrics + envs) lives under this S3
# prefix. ONLY the eval pod's credentials may read it; research pods cannot.
SECRET_DATA_PREFIX = os.getenv("SECRET_DATA_PREFIX", "holdout/")
# Where research pods upload a submitted model, and where the eval pod writes
# scores.json back. Both under the run's own prefix (research-writable).
SUBMISSION_PREFIX = os.getenv("SUBMISSION_PREFIX", "submissions/")

# Ephemeral eval pod (spawned per submission; scores then self-terminates).
EVAL_POD_TEMPLATE_ID = os.getenv("EVAL_POD_TEMPLATE_ID", "")
EVAL_POD_GPU_TYPE = os.getenv("EVAL_POD_GPU_TYPE", "NVIDIA H200")
EVAL_POD_GPU_COUNT = int(os.getenv("EVAL_POD_GPU_COUNT", "1"))
EVAL_POD_TIMEOUT_SECONDS = int(os.getenv("EVAL_POD_TIMEOUT_SECONDS", str(2 * 3600)))
EVAL_POLL_INTERVAL_SECONDS = int(os.getenv("EVAL_POLL_INTERVAL_SECONDS", "30"))
# OpenAI key for judge-category benchmarks (read only on the eval side).
OAI_API_KEY = os.getenv("OAI_API", os.getenv("OPENAI_API_KEY", ""))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")

# =============================================================================
# Transport / isolation
# =============================================================================
# How the trained model + scores move between the research side and the eval
# side, and where the secret suite lives.
#   "fs" (default) — shared filesystem (VAST). Isolation = file permissions:
#                    HOLDOUT_DIR is owned by a separate eval user, mode 700, so
#                    the AAR (running as you) can neither read nor DELETE it.
#   "s3"           — S3-compatible object store (for infra with no shared FS,
#                    e.g. cross-data-center RunPod pods). Isolation = creds.
HARNESS_TRANSPORT = os.getenv("HARNESS_TRANSPORT", "fs")

# Hardened two-user flow: a persistent eval_worker.sh (running as the SECRET-
# holding user, e.g. <you>-eval) drains the submission queue and scores models
# against the mode-700 holdout. When true (default), the research side (the AAR)
# must NOT launch its own eval — it cannot read the holdout — so evaluate_model
# only stages the model and POLLS for the worker's scores. Set false only for a
# legacy single-user dev box where the same user both trains and evals in one job.
EVAL_VIA_WORKER = os.getenv("EVAL_VIA_WORKER", "true").lower() in ("1", "true", "yes")

# Shared-FS layout (kept OUTSIDE any AAR worktree / results dir so AAR cleanup
# can never touch the holdout). The holdout is also reproducible from a publish
# script + public data; back up the answer keys to cold storage (see ISOLATION.md).
HARNESS_RUNS_DIR = os.getenv("HARNESS_RUNS_DIR", f"{WORKSPACE_DIR}/_runs")
# TEAM_DIR — the single per-team home. When set (by launch_team.sh), EVERY mutable
# team artifact (forum, submissions, scores, generated methods, logs, the team's own
# in-run literature) defaults to a subdir of it: aar_teams/<TEAM_ID>/{forum,submissions,
# scores,methods,logs,litreview}. Explicit per-dir env still wins; when TEAM_DIR is
# unset we fall back to the legacy shared layout. Exception (set separately): the
# axis-wise literature baseline lives per-AXIS, not under TEAM_DIR.
TEAM_DIR = os.getenv("TEAM_DIR", "").strip()
def _team_sub(name: str, fallback: str) -> str:
    return f"{TEAM_DIR}/{name}" if TEAM_DIR else fallback
# Two cross-user channels (group `shared`, setgid). Research writes models to
# SUBMISSIONS_DIR; the eval user writes scores to SCORES_DIR. Split so neither
# side writes into the other's dirs — only group-reads — keeping permissions simple.
SUBMISSIONS_DIR = os.getenv("SUBMISSIONS_DIR") or _team_sub("submissions", f"{HARNESS_RUNS_DIR}/submissions")
SCORES_DIR = os.getenv("SCORES_DIR") or _team_sub("scores", f"{HARNESS_RUNS_DIR}/scores")
# The SECRET suite root — read ONLY by the eval user's jobs. In the hardened
# setup this is the eval user's mode-700 path (e.g. /opt/aar/work
# holdout), which the research user (the AAR) cannot read. Dev fallback: your
# own space (honor system). See ISOLATION.md.
HOLDOUT_DIR = os.getenv("HOLDOUT_DIR", f"{WORKSPACE_DIR}/_holdout")
# EVAL-PRIVATE held-out scores. The held-out (generalization) benchmark's FULL score
# is written here by the eval worker (mode-700, eval-user-owned — NOT in research-
# readable SCORES_DIR), so the AAR can never read it even via the filesystem; the
# human reads the generalization check here with eval credentials. run_eval --heldout-dir.
HELDOUT_SCORES_DIR = os.getenv("HELDOUT_SCORES_DIR", f"{HOLDOUT_DIR}/heldout_scores")
# GOLDEN eval configs. Each benchmark's decoding (batch / auto_ceiling / no_repeat_ngram /
# max_new_tokens) lives in benchmark_docs/<group>/baseline.json — the SAME config its
# baseline was measured at. The eval reads it PER-BENCHMARK so a per-iteration eval that
# bundles benchmarks from different baseline groups (e.g. the shared capability basket at
# batch8/ceiling4096/ngram0 alongside an axis's safety benches at the axis's decoding)
# scores each one exactly as its baseline was — not at one global per-run setting. Lives
# eval-side (deployed by publish_holdout); overridable for local/dev.
BENCHMARK_DOCS_DIR = os.getenv("BENCHMARK_DOCS_DIR", f"{WORKSPACE_DIR}/benchmark_docs")
# Slurm script the orchestrator submits to run an eval (fs transport).
EVAL_SLURM_SCRIPT = os.getenv("EVAL_SLURM_SCRIPT", f"{WORKSPACE_DIR}/scripts/slurm_eval.sh")

# =============================================================================
# S3 / AWS (required for RunPod mode, unused in local mode)
# =============================================================================

S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_IDEAS_PREFIX = os.getenv("S3_IDEAS_PREFIX", "ideas/")
S3_RESULTS_PREFIX = os.getenv("S3_RESULTS_PREFIX", "results/")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# =============================================================================
# Server (orchestrator API for AAR evaluation)
# =============================================================================

SERVER_URL = os.getenv("ORCHESTRATOR_API_URL", "http://localhost:8000")
AAR_MODE = os.getenv("AAR_MODE", "true").lower() in ("1", "true", "yes")
# Local mode = no central orchestrator server; chains coordinate over the shared
# filesystem. Set by the cluster launcher (slurm_aar_chain.sh).
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() in ("1", "true", "yes")
# Forum backend for the share_finding / get_leaderboard tools:
#   "fs"   — the shared findings dir IS the forum (default in local mode). No
#            server; every chain reads/writes LOCAL_FINDINGS_DIR on shared VAST.
#   "http" — POST/GET a central orchestrator (needs `python run.py server` up and
#            ORCHESTRATOR_API_URL reachable from every chain).
FORUM_BACKEND = os.getenv("FORUM_BACKEND", "fs" if LOCAL_MODE else "http")

# =============================================================================
# Agent loop
# =============================================================================

FULL_AUTO_MAX_RUNTIME_SECONDS = int(os.getenv("FULL_AUTO_MAX_RUNTIME_SECONDS", str(5 * 24 * 3600)))
# Per-session transcripts (SESSION_LOGS_DIR alias kept for the dashboard) and the
# team's findings forum both default under TEAM_DIR when set.
LOGS_DIR = os.getenv("SESSION_LOGS_DIR") or os.getenv("LOGS_DIR") or _team_sub("logs", f"{WORKSPACE_DIR}/aar/research_loop/logs")
LOCAL_FINDINGS_DIR = os.getenv("LOCAL_FINDINGS_DIR") or _team_sub("forum", f"{WORKSPACE_DIR}/aar/research_loop/shared_findings")
# IDEAS_DIR — where THIS team's generated method packages (aar.ideas-style <name>/run.py)
# live. Under TEAM_DIR/methods when set; the repo's aar/ideas/ otherwise. The seed
# library (TEMPLATE + the axis seed) always remains readable in the repo aar/ideas/ as
# a fallback (see _resolve_idea_dir users).
IDEAS_DIR = os.getenv("AAR_IDEAS_DIR") or _team_sub("methods", f"{WORKSPACE_DIR}/aar/ideas")
# The repo's seed-method library — always available as a read fallback for seeds.
SEED_IDEAS_DIR = f"{WORKSPACE_DIR}/aar/ideas"
def resolve_idea_dir(name: str):
    """Path to a method package <name>/: the team's IDEAS_DIR first, then the repo
    seed library. Returns the team path (may not exist) when neither is present, so
    callers can still report a sensible missing-path. Sanitizes against traversal."""
    from pathlib import Path as _P
    safe = "".join(c for c in (name or "") if c.isalnum() or c in "-_")
    if not safe:
        return _P(IDEAS_DIR) / "_invalid_"
    team = _P(IDEAS_DIR) / safe
    if team.is_dir():
        return team
    seed = _P(SEED_IDEAS_DIR) / safe
    if seed.is_dir():
        return seed
    return team
FINDINGS_POLL_INTERVAL = int(os.getenv("FINDINGS_POLL_INTERVAL", "60"))
TARGET_IDEA_FILE = f"{WORKSPACE_DIR}/aar/research_loop/target_idea/idea.json"

# =============================================================================
# RunPod deployment (server-side, for spawning worker pods)
# =============================================================================

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_TEMPLATE_ID = os.getenv("RUNPOD_TEMPLATE_ID", "")
RUNPOD_GPU_TYPE = os.getenv("RUNPOD_GPU_TYPE", "NVIDIA H200")
DEPLOY_TO_RUNPOD = os.getenv("DEPLOY_TO_RUNPOD", "false").lower() == "true"

MAX_CONCURRENT_PODS = int(os.getenv("MAX_CONCURRENT_PODS", "1"))
POD_DEPLOY_MAX_RETRIES = int(os.getenv("POD_DEPLOY_MAX_RETRIES", "100000000"))
POD_DEPLOY_RETRY_DELAY_SECONDS = int(os.getenv("POD_DEPLOY_RETRY_DELAY_SECONDS", "300"))
FULL_AUTO_WORKER_MAX_RUNTIME_SECONDS = int(os.getenv("FULL_AUTO_WORKER_MAX_RUNTIME_SECONDS", str(5 * 24 * 3600)))
FULL_AUTO_POD_TIMEOUT_SECONDS = int(os.getenv("FULL_AUTO_POD_TIMEOUT_SECONDS", str(6 * 24 * 3600)))

# =============================================================================
# Docker local mode (isolated container with GPU, no labels inside)
# =============================================================================

DOCKER_LOCAL_MODE = os.getenv("DOCKER_LOCAL_MODE", "false").lower() == "true"
DOCKER_LOCAL_IMAGE = os.getenv("DOCKER_LOCAL_IMAGE", "w2s-research")
DOCKER_EXECUTABLE = os.getenv("DOCKER_EXECUTABLE", "docker")

# =============================================================================
# Training defaults
# =============================================================================

EPOCHS = 5
NUM_GPUS = 5
