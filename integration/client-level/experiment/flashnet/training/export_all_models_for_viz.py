#!/usr/bin/env python3
"""
Find all model.keras files under data/, export each to JSON weights, and create a manifest
for the visualization dropdown.

Usage:
  python export_all_models_for_viz.py [--data-dir path] [--output-dir path]
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Add parent for import
sys.path.insert(0, str(Path(__file__).parent))
from export_weights_for_viz import export_weights


def path_to_label(model_path: Path, data_root: Path) -> str:
    """Create a short human-readable label from the model path."""
    try:
        rel = model_path.relative_to(data_root)
        parts = rel.parts
        # e.g. alibaba.../.../nvme0n1...nvme1n1/flashnet/training_results/mldrive0/nnK/model.keras
        # Extract: trace profile, device pair, mldrive
        label_parts = []
        for i, p in enumerate(parts):
            if "nvme" in p and "..." in p:
                label_parts.append(p)
            elif p in ("mldrive0", "mldrive1"):
                label_parts.append(p)
            elif i == 0 and len(parts) > 3:
                # First part often has trace name
                short = p[:30] + ".." if len(p) > 32 else p
                label_parts.append(short)
        return " / ".join(label_parts[-3:]) if label_parts else rel.stem
    except ValueError:
        return model_path.stem


def find_dataset(model_path: Path) -> Path | None:
    """Find the training dataset CSV for a model (mldrive0.csv or mldrive1.csv)."""
    # model_path: .../training_results/mldrive0/nnK/model.keras
    # training_results = model_path.parent.parent.parent (nnK -> mldrive0 -> training_results)
    training_results = model_path.parent.parent.parent
    path_str = str(model_path)
    if "/mldrive0/" in path_str:
        dataset = training_results / "mldrive0.csv"
    elif "/mldrive1/" in path_str:
        dataset = training_results / "mldrive1.csv"
    else:
        return None
    return dataset if dataset.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).parent.parent.parent.parent / "data"),
        help="Root data directory to search for model.keras",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent.parent / "weights"),
        help="Output directory for JSON weights and manifest",
    )
    parser.add_argument(
        "--train-eval-split",
        default="80_20",
        help="Train/eval split for scaler",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_files = sorted(data_dir.rglob("model.keras"))
    # Only flashnet nnK models
    model_files = [p for p in model_files if "flashnet" in str(p) and "nnK" in str(p)]

    if not model_files:
        print(f"No model.keras files found under {data_dir}")
        sys.exit(1)

    manifest = {"models": []}
    for i, model_path in enumerate(model_files):
        dataset_path = find_dataset(model_path)
        if not dataset_path:
            print(f"SKIP (no dataset): {model_path}")
            continue

        label = path_to_label(model_path, data_dir)
        safe_id = re.sub(r"[^\w\-]", "_", str(model_path.relative_to(data_dir)))[:80]
        out_file = f"weights_{i:03d}.json"
        out_path = output_dir / out_file

        try:
            export_weights(str(model_path), str(dataset_path), str(out_path), args.train_eval_split)
            manifest["models"].append({
                "id": out_file.replace(".json", ""),
                "label": label,
                "file": out_file,
            })
        except Exception as e:
            print(f"ERROR exporting {model_path}: {e}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path} ({len(manifest['models'])} models)")

    return manifest


if __name__ == "__main__":
    main()
