#!/usr/bin/env python3
"""
Export Heimdall/FlashNet model weights and scaler to JSON for the interactive visualization.
Run this after training a model to use real weights in the visualization.

Usage:
  python export_weights_for_viz.py -model path/to/model.keras -dataset path/to/dataset.csv -output weights.json
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers


INPUT_FEATURES = [
    "queue_len", "io_type", "size",
    "prev_queue_len_1", "prev_queue_len_2", "prev_queue_len_3",
    "prev_latency_1", "prev_latency_2", "prev_latency_3",
    "prev_throughput_1", "prev_throughput_2", "prev_throughput_3",
]


def load_model_weights(model_path: Path):
    """Load weights from .keras by building architecture and loading from zip (avoids deserialization issues)."""
    model = keras.Sequential([
        layers.Dense(128, input_dim=12, name="dense_1"),
        layers.Activation("relu", name="relu_1"),
        layers.Dense(16, name="dense_2"),
        layers.Activation("relu", name="relu_2"),
        layers.Dense(1, name="dense_3"),
        layers.Activation("sigmoid", name="sigmoid_out"),
    ])
    model_path = Path(model_path)
    with zipfile.ZipFile(model_path, "r") as zf:
        if "model.weights.h5" not in zf.namelist():
            raise FileNotFoundError(f"No model.weights.h5 in {model_path}")
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(zf.read("model.weights.h5"))
            tmp_path = tmp.name
    try:
        model.load_weights(tmp_path, by_name=True, skip_mismatch=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return model


def export_weights(model_path: str, dataset_path: str, output_path: str, train_eval_split: str = "80_20"):
    """Export model weights and scaler to JSON for browser visualization."""
    model = load_model_weights(Path(model_path))
    dataset = pd.read_csv(dataset_path)

    # Match training preprocessing
    reordered_cols = [col for col in dataset.columns if col != "latency"] + ["latency"]
    dataset = dataset[reordered_cols]
    x = dataset.copy(deep=True).drop(columns=["reject"], axis=1)
    y = dataset["reject"]

    from sklearn.model_selection import train_test_split
    ratios = train_eval_split.split("_")
    percent_eval = int(ratios[1])
    x_train, _, y_train, _ = train_test_split(x, y, test_size=percent_eval / 100, random_state=42)
    x_train = x_train.drop(columns=["latency"], axis=1)

    # Ensure feature order matches INPUT_FEATURES
    missing = [c for c in INPUT_FEATURES if c not in x_train.columns]
    if missing:
        print(f"WARNING: Missing features {missing}; using 0 for scaler fit")
    for c in INPUT_FEATURES:
        if c not in x_train.columns:
            x_train[c] = 0
    x_train = x_train[INPUT_FEATURES]

    scaler = MinMaxScaler()
    scaler.fit(x_train)

    export = {
        "input_features": INPUT_FEATURES,
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_scale": scaler.data_range_.tolist(),
        "layers": [],
    }

    # Dense layers are at indices 0, 2, 4
    for idx in [0, 2, 4]:
        layer = model.layers[idx]
        w, b = layer.get_weights()
        export["layers"].append({
            "name": layer.name,
            "weights": w.tolist(),
            "bias": b.tolist(),
        })

    with open(output_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"Exported weights to {output_path}")
    return export


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", required=True, help="Path to model.keras")
    parser.add_argument("-dataset", required=True, help="Path to training dataset CSV")
    parser.add_argument("-output", default="heimdall_weights.json", help="Output JSON path")
    parser.add_argument("-train_eval_split", default="80_20", help="e.g. 80_20")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.dataset):
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)

    export_weights(args.model, args.dataset, args.output, args.train_eval_split)
