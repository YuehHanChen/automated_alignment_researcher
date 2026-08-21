"""Transport layer: how the trained model + scores move between the research
side and the eval side, and where the secret suite is read from.

Two backends, selected by config.HARNESS_TRANSPORT:

- "fs" (default) — shared filesystem (VAST). Model is copied under
  SUBMISSIONS_DIR/<run_id>/model/; the eval reads it + the secret suite from
  HOLDOUT_DIR in place (no copy); scores are written back next to the model.
  Isolation = file permissions (HOLDOUT_DIR owned by a separate eval user,
  mode 700), which also prevents accidental deletion by the AAR.

- "s3" — S3-compatible object store, for infra with no shared filesystem.
  Isolation = scoped credentials.

Everything else in the harness calls these functions and stays
transport-agnostic.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from aar import config


def _fs() -> bool:
    return config.HARNESS_TRANSPORT == "fs"


def _copytree_retry(src, dst, attempts: int = 4):
    """copytree that retries transient shared-FS errors (NFS 'Stale file handle'
    [Errno 116] / Errno 521 seen on the VAST mount during the train->submissions
    model handoff). A transient FS error must not lose a finished training run; we
    back off and retry, cleaning a partial dest each time, before giving up."""
    last = None
    for i in range(attempts):
        try:
            shutil.copytree(src, dst)
            return
        except (OSError, shutil.Error) as e:
            last = e
            shutil.rmtree(dst, ignore_errors=True)
            if i < attempts - 1:
                time.sleep(2 ** i)   # 1,2,4s
    raise last


# --- model handoff (research -> eval) --------------------------------------
def put_model(model_path: str, run_id: str) -> str:
    """Research side: publish the trained model for run_id. Returns its ref."""
    if _fs():
        dest = Path(config.SUBMISSIONS_DIR) / run_id / "model"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        _copytree_retry(model_path, dest)
        # Atomicity marker: the eval worker only picks up run_ids once this exists
        # (so it never scores a half-copied model).
        (dest.parent / ".submitted").touch()
        return str(dest)
    from aar.infrastructure import s3_utils
    prefix = f"{config.SUBMISSION_PREFIX}{run_id}/model"
    s3_utils.upload_directory_to_s3(Path(model_path), prefix, config.S3_BUCKET)
    return f"s3://{config.S3_BUCKET}/{prefix}"


def get_model(run_id: str, dest_dir: Optional[str] = None) -> str:
    """Eval side: obtain a local path to the submitted model."""
    if _fs():
        return str(Path(config.SUBMISSIONS_DIR) / run_id / "model")
    from aar.infrastructure import s3_utils
    dest = Path(dest_dir or f"./eval_work/{run_id}/model")
    dest.mkdir(parents=True, exist_ok=True)
    s3_utils.download_s3_directory(dest, config.S3_BUCKET, f"{config.SUBMISSION_PREFIX}{run_id}/model/", force_download=True, description="model")
    return str(dest)


# --- secret suite (eval side reads only) -----------------------------------
def resolve_suite_dir(suite: str, dest_dir: Optional[str] = None) -> str:
    """Eval side: local path to the secret suite dir (test inputs + answers +
    <suite>.yaml). FS: read HOLDOUT_DIR in place. S3: download from secret prefix."""
    if _fs():
        return str(Path(config.HOLDOUT_DIR) / suite)
    from aar.infrastructure import s3_utils
    dest = Path(dest_dir or f"./eval_work/secret/{suite}")
    dest.mkdir(parents=True, exist_ok=True)
    s3_utils.download_s3_directory(dest, config.S3_BUCKET, f"{config.SECRET_DATA_PREFIX}{suite}/", force_download=True, description="secret suite")
    return str(dest)


# --- scores (eval -> research/orchestrator) --------------------------------
def put_scores(run_id: str, scores: dict[str, Any]) -> str:
    """Eval side: publish scores for run_id (the completion signal). Written to
    the SCORES_DIR channel (eval-writable, research-readable), so the eval user
    never writes into the research user's submission dirs."""
    payload = json.dumps(scores, indent=2)
    if _fs():
        path = Path(config.SCORES_DIR) / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
        return str(path)
    from aar.infrastructure import s3_utils
    tmp = Path(f"./eval_work/{run_id}/scores.json")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(payload)
    s3_utils.upload_file_to_s3(tmp, f"{config.SUBMISSION_PREFIX}{run_id}/scores.json", config.S3_BUCKET, content_type="application/json")
    return f"s3://{config.S3_BUCKET}/{config.SUBMISSION_PREFIX}{run_id}/scores.json"


def read_scores(run_id: str) -> Optional[dict[str, Any]]:
    """Research/orchestrator side: return scores if present yet, else None."""
    if _fs():
        path = Path(config.SCORES_DIR) / f"{run_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None
    from aar.infrastructure import s3_utils
    s3 = s3_utils.get_s3_client()
    try:
        obj = s3.get_object(Bucket=config.S3_BUCKET, Key=f"{config.SUBMISSION_PREFIX}{run_id}/scores.json")
        return json.loads(obj["Body"].read())
    except Exception:
        return None
