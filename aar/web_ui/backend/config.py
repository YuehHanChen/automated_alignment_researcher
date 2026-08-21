"""
Flask-specific configuration for the web UI backend.

All shared settings (paths, models, S3, RunPod, etc.) live in
aar.config. This file only adds Flask/DB settings and
derived paths specific to the server process.
"""
import os
from pathlib import Path

from aar.config import *  # noqa: F401,F403
from aar.config import WORKSPACE_DIR, DATASET_NAME, SERVER_URL

# =============================================================================
# Server paths
# =============================================================================

BASE_DIR = Path(__file__).parent
DATABASE_PATH = BASE_DIR / "experiments.db"
RESULTS_DIR = Path(WORKSPACE_DIR) / "results"
SERVER_CACHE_ROOT = os.getenv("SERVER_CACHE_ROOT", str(Path(WORKSPACE_DIR) / "cache_results"))

DATA_BASE_DIR = f"{WORKSPACE_DIR}/data"
DATASET_DIR = f"{DATA_BASE_DIR}/{DATASET_NAME}"
ORCHESTRATOR_API_URL = os.getenv("ORCHESTRATOR_API_URL", SERVER_URL)

# =============================================================================
# Flask / Database
# =============================================================================

SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
SQLALCHEMY_TRACK_MODIFICATIONS = False
POLL_INTERVAL = 5

# =============================================================================
# Web UI dropdowns
# =============================================================================

AVAILABLE_DATASETS = ["chat"]
AVAILABLE_WEAK_MODELS = ["Qwen/Qwen1.5-0.5B-Chat"]
AVAILABLE_STRONG_MODELS = ["Qwen/Qwen3-4B-Base"]

# =============================================================================
# Misc server settings
# =============================================================================

SKIP_PRIOR_WORK_SEARCH = False
MAX_IMPROVEMENT_ITERATIONS = 10
