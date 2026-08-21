#!/usr/bin/env python3
"""
Convert labeled_data/ to data/ by removing train_label.jsonl and stripping
metadata fields from test.jsonl and train_unlabel.jsonl.

Only keeps the fields needed for training/evaluation. Strips labels, difficulty,
subset, and other metadata.

Usage:
    python scripts/prepare_data.py --labeled-dir labeled_data --output-dir data
"""
import argparse
import json
import os
from pathlib import Path


STRIP_FIELDS = {"label", "subset", "difficulty"}


def strip_labels(input_path: Path, output_path: Path):
    """Copy a JSONL file, removing metadata fields from each row."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            for field in STRIP_FIELDS:
                row.pop(field, None)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert labeled_data/ to data/ (strip labels)")
    parser.add_argument("--labeled-dir", default="labeled_data", help="Source directory with labels")
    parser.add_argument("--output-dir", default="data", help="Output directory without labels")
    args = parser.parse_args()

    labeled_dir = Path(args.labeled_dir)
    output_dir = Path(args.output_dir)

    if not labeled_dir.exists():
        print(f"Error: {labeled_dir} not found")
        return

    datasets = sorted(d.name for d in labeled_dir.iterdir() if d.is_dir())
    if not datasets:
        print(f"Error: no dataset subdirectories found in {labeled_dir}")
        return

    print(f"Converting {labeled_dir}/ -> {output_dir}/")
    print(f"Datasets: {', '.join(datasets)}\n")

    for dataset in datasets:
        src = labeled_dir / dataset
        dst = output_dir / dataset
        dst.mkdir(parents=True, exist_ok=True)

        for filename in ["test.jsonl", "train_unlabel.jsonl"]:
            src_file = src / filename
            if not src_file.exists():
                print(f"  Warning: {src_file} not found, skipping")
                continue
            n = strip_labels(src_file, dst / filename)
            print(f"  {dataset}/{filename}: {n} rows (label field removed)")

        # Skip train_label.jsonl (not copied to output)
        print(f"  {dataset}/train_label.jsonl: skipped (labels withheld)\n")

    print("Done.")


if __name__ == "__main__":
    main()
