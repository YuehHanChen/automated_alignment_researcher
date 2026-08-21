"""
Server-side evaluation for AAR (Automated Alignment Research).

This module computes metrics from worker pod predictions using locally
stored ground truth labels. Workers never have access to labels - they
only upload raw predictions.

Usage:
    from aar.web_ui.backend.evaluation import (
        compute_metrics_from_predictions,
        load_ground_truth_labels,
    )
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import numpy as np


# Default ground truth directory (labeled_data contains JSONL files with labels)
DEFAULT_GROUND_TRUTH_DIR = os.getenv("GROUND_TRUTH_DIR", os.path.join(os.getenv("WORKSPACE_DIR", "."), "labeled_data"))


def load_ground_truth_labels(
    dataset_name: str,
    split: str = "test",
    ground_truth_dir: Optional[str] = None,
) -> List[int]:
    """
    Load ground truth labels from server-only storage.
    
    Supports two formats:
    1. JSONL files in labeled_data/{dataset}/{split}.jsonl (primary)
    2. JSON files in ground_truth/{dataset}/{split}_labels.json (legacy)
    
    Args:
        dataset_name: Name of dataset (e.g., "math-binary", "chat-0115")
        split: Which split to load labels for ("test" or "train_unlabel")
        ground_truth_dir: Path to ground truth directory
        
    Returns:
        List of ground truth labels (0 or 1)
        
    Raises:
        FileNotFoundError: If labels file doesn't exist
        ValueError: If invalid split specified
    """
    if ground_truth_dir is None:
        ground_truth_dir = os.getenv("GROUND_TRUTH_DIR", DEFAULT_GROUND_TRUTH_DIR)
    
    if split not in ("test", "train_unlabel"):
        raise ValueError(f"Invalid split: {split}. Must be 'test' or 'train_unlabel'")
    
    ground_truth_path = Path(ground_truth_dir)
    
    # Try JSONL format first (primary format in labeled_data/)
    jsonl_file = ground_truth_path / dataset_name / f"{split}.jsonl"
    if jsonl_file.exists():
        labels = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Handle both numeric labels and string labels (first/second format)
                    label = item.get("label")
                    if isinstance(label, str):
                        # Convert 'first'/'second' or 'A'/'B' to 0/1
                        label = 0 if label.lower() in ('first', 'a', '0') else 1
                    labels.append(int(label))
        return labels
    
    # Fallback to JSON format (legacy ground_truth/ format)
    json_file = ground_truth_path / dataset_name / f"{split}_labels.json"
    if json_file.exists():
        with open(json_file, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(
        f"Ground truth labels not found for dataset '{dataset_name}', split '{split}'.\n"
        f"Checked paths:\n"
        f"  - {jsonl_file}\n"
        f"  - {json_file}\n"
        f"Ensure labeled data exists in {ground_truth_dir}/{dataset_name}/"
    )


def compute_metrics_from_predictions(
    predictions: List[int],
    ground_truth: List[int],
    fixed_weak_acc: Optional[float] = None,
    fixed_strong_acc: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics from predictions and ground truth labels.
    
    This is the SERVER-SIDE function for AAR. It takes predictions from
    worker pods and computes metrics using locally stored ground truth.
    
    Args:
        predictions: List of predicted labels (0 or 1) from worker pod
        ground_truth: List of ground truth labels (0 or 1) from server storage
        fixed_weak_acc: Fixed weak baseline accuracy (for PGR computation)
        fixed_strong_acc: Fixed strong/ceiling baseline accuracy (for PGR computation)
        
    Returns:
        Dictionary with:
        - transfer_acc: Transfer accuracy (predictions vs ground truth)
        - pgr: Performance Gap Recovery (if baselines provided)
        - correct: Number of correct predictions
        - total: Total number of predictions
        - pred_distribution: Distribution of predictions
        - label_distribution: Distribution of ground truth labels
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Predictions ({len(predictions)}) and ground truth ({len(ground_truth)}) "
            f"must have the same length"
        )
    
    if len(predictions) == 0:
        return {
            "transfer_acc": 0.0,
            "pgr": None,
            "correct": 0,
            "total": 0,
            "pred_distribution": {},
            "label_distribution": {},
        }
    
    # Convert to numpy for efficient computation
    predictions_arr = np.array(predictions)
    ground_truth_arr = np.array(ground_truth)
    
    # Compute accuracy
    correct = int((predictions_arr == ground_truth_arr).sum())
    total = len(predictions)
    transfer_acc = correct / total
    
    # Compute PGR if baselines are provided
    pgr = None
    if fixed_weak_acc is not None and fixed_strong_acc is not None:
        if fixed_strong_acc > fixed_weak_acc:
            pgr = (transfer_acc - fixed_weak_acc) / (fixed_strong_acc - fixed_weak_acc)
        else:
            # Edge case: strong not better than weak (shouldn't happen normally)
            pgr = None
    
    # Compute distributions
    pred_distribution = {int(k): int(v) for k, v in Counter(predictions).items()}
    label_distribution = {int(k): int(v) for k, v in Counter(ground_truth).items()}
    
    return {
        "transfer_acc": transfer_acc,
        "pgr": pgr,
        "correct": correct,
        "total": total,
        "pred_distribution": pred_distribution,
        "label_distribution": label_distribution,
        "fixed_weak_acc": fixed_weak_acc,
        "fixed_strong_acc": fixed_strong_acc,
    }


def get_fixed_baselines(
    dataset_name: str,
    weak_model: Optional[str] = None,
    strong_model: Optional[str] = None,
    cache_root: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get fixed weak and strong baselines for a dataset.
    
    This retrieves pre-computed baselines that are used consistently
    across all experiments for PGR calculation.
    
    Args:
        dataset_name: Name of dataset
        weak_model: Weak model name (for lookup)
        strong_model: Strong model name (for lookup)
        cache_root: Optional cache root directory (default: uses HierarchicalCache default)
        
    Returns:
        Tuple of (fixed_weak_acc, fixed_strong_acc)
        Either may be None if not available
    """
    # Try to load from hierarchical cache
    try:
        from aar.utils import (
            get_fixed_weak_baseline,
            get_fixed_ceiling_baseline,
        )
        from pathlib import Path
        
        fixed_weak_acc = None
        fixed_strong_acc = None
        
        # Log cache root being used
        # Use SERVER_CACHE_ROOT from config if cache_root not explicitly provided
        from aar.web_ui.backend import config
        actual_cache_root = cache_root or config.SERVER_CACHE_ROOT
        cache_path = Path(actual_cache_root)
        if not cache_path.exists():
            print(f"[get_fixed_baselines] WARNING: Cache root does not exist: {actual_cache_root}")
        else:
            print(f"[get_fixed_baselines] Searching cache at: {actual_cache_root}")
        
        if weak_model:
            weak_result = get_fixed_weak_baseline(
                weak_model=weak_model,
                dataset_name=dataset_name,
                cache_root=cache_root,
            )
            if weak_result and weak_result[0] is not None:
                fixed_weak_acc = weak_result[0]
                print(f"[get_fixed_baselines] Found weak baseline: {fixed_weak_acc:.4f} (from {weak_result[2]} seeds)")
            else:
                print(f"[get_fixed_baselines] Weak baseline not found for {weak_model} on {dataset_name}")
        
        if strong_model:
            strong_result = get_fixed_ceiling_baseline(
                strong_model=strong_model,
                dataset_name=dataset_name,
                cache_root=cache_root,
            )
            if strong_result and strong_result[0] is not None:
                fixed_strong_acc = strong_result[0]
                print(f"[get_fixed_baselines] Found strong baseline: {fixed_strong_acc:.4f} (from {strong_result[2]} seeds)")
            else:
                print(f"[get_fixed_baselines] Strong baseline not found for {strong_model} on {dataset_name}")
        
        if fixed_weak_acc is None or fixed_strong_acc is None:
            print(f"[get_fixed_baselines] Missing baselines - weak: {fixed_weak_acc}, strong: {fixed_strong_acc}")
            print(f"[get_fixed_baselines] Cache root: {actual_cache_root}")
            if cache_path.exists():
                # List what's in the cache
                dataset_weak_dir = cache_path / f"{dataset_name}_weak_artifacts"
                dataset_ceiling_dir = cache_path / f"{dataset_name}_train_ceiling"
                print(f"[get_fixed_baselines] Checking directories:")
                print(f"  - {dataset_weak_dir} exists: {dataset_weak_dir.exists()}")
                print(f"  - {dataset_ceiling_dir} exists: {dataset_ceiling_dir.exists()}")
                if dataset_weak_dir.exists():
                    subdirs = list(dataset_weak_dir.iterdir())
                    print(f"  - Found {len(subdirs)} subdirectories in weak_artifacts")
                if dataset_ceiling_dir.exists():
                    subdirs = list(dataset_ceiling_dir.iterdir())
                    print(f"  - Found {len(subdirs)} subdirectories in train_ceiling")
        
        return fixed_weak_acc, fixed_strong_acc
        
    except ImportError:
        # Cache functions not available
        print("[get_fixed_baselines] ERROR: Cache functions not available (ImportError)")
        return None, None
    except Exception as e:
        print(f"[get_fixed_baselines] ERROR: Exception while getting baselines: {e}")
        import traceback
        traceback.print_exc()
        return None, None
