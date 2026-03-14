#!/usr/bin/env python3
"""
Scan activation data in the data/ folder and compute min/max for the visualization color scale.
Outputs activation_range.json with scale_min and scale_max (post-ReLU: min=0, max from data).

Usage:
  python compute_activation_range.py [--data-dir path] [--output path]
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def compute_range(data_dir: Path, output_path: Path, sample_rate: int = 1) -> tuple[float, float]:
    """Scan all_activations CSV files and compute post-ReLU min/max."""
    all_vals = []
    files_processed = 0

    for f in sorted(data_dir.rglob("*all_activations*.csv")):
        if files_processed % sample_rate != 0:
            files_processed += 1
            continue
        try:
            with open(f) as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    try:
                        v = float(row.get("activation_value", 0))
                        all_vals.append(max(0, v))  # post-ReLU
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
        files_processed += 1

    if not all_vals:
        return 0.0, 1.0

    scale_min = min(all_vals)
    scale_max = max(all_vals)
    if scale_max <= scale_min:
        scale_max = scale_min + 1.0

    result = {"scale_min": scale_min, "scale_max": scale_max, "n_values": len(all_vals)}
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Processed {len(all_vals)} values from {files_processed} files")
    print(f"Range: [{scale_min}, {scale_max}] -> {output_path}")
    return scale_min, scale_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="integration/client-level/data",
        help="Root data directory to scan",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: same dir as this script)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=100,
        help="Process every Nth file (1=all, 100=sample 1%%)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data dir not found: {data_dir}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path(__file__).parent.parent / "activation_range.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compute_range(data_dir, output_path, args.sample_rate)
