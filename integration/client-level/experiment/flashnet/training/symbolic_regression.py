#!/usr/bin/env python3
"""
Symbolic regression to mimic the neural network learned in each data directory.

For each flashnet/training_results directory containing a trained nnK model,
this script:
1. Loads the model and dataset
2. Gets NN predictions (probabilities) on the data
3. Runs symbolic regression (gplearn SymbolicClassifier) to find a mathematical
   expression that predicts 0 or 1, mimicking the NN's binary decisions
4. Saves the symbolic expression and metrics to the directory

Usage:
  python symbolic_regression.py [data_directory]
  python symbolic_regression.py -model path/to/model.keras -dataset path/to/mldrive0.csv

Output (per model):
  - symbolic_regression_expression.txt: The discovered formula (output thresholded to 0/1)
  - symbolic_regression_metrics.json: accuracy, agreement with NN, R², correlation
  - symbolic_regression_report.md: Human-readable report
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

# Add script directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def find_model_dirs(data_dir: Path) -> list[tuple[Path, Path, str]]:
    """
    Find all (model_path, dataset_path, mldrive_name) tuples under data_dir.
    Returns list of tuples for each mldrive0/nnK and mldrive1/nnK model.
    """
    results = []
    for model_path in data_dir.rglob("model.keras"):
        if "flashnet" not in str(model_path) or "nnK" not in str(model_path):
            continue
        # model_path: .../training_results/mldrive0/nnK/model.keras
        training_results = model_path.parent.parent.parent  # training_results dir
        mldrive_name = model_path.parent.parent.name  # mldrive0 or mldrive1
        dataset_path = training_results / f"{mldrive_name}.csv"
        if dataset_path.exists():
            results.append((model_path, dataset_path, mldrive_name))
    return sorted(results, key=lambda x: str(x[0]))


def load_model_compat(model_path: Path):
    """
    Load nnK model by building architecture and loading weights.
    Avoids Keras deserialization issues with batch_input_shape.
    """
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


def load_data_and_model(model_path: Path, dataset_path: Path, train_eval_split: str = "50_50"):
    """
    Load model and dataset with same preprocessing as nnK.py.
    Returns (X_norm, y_nn_pred, scaler, feature_names).
    """
    ratios = train_eval_split.split("_")
    percent_eval = int(ratios[1])
    assert int(ratios[0]) + percent_eval == 100

    model = load_model_compat(model_path)
    dataset = pd.read_csv(dataset_path)

    reordered_cols = [col for col in dataset.columns if col != "latency"] + ["latency"]
    dataset = dataset[reordered_cols]

    x = dataset.drop(columns=["reject"], axis=1)
    y = dataset["reject"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=percent_eval / 100, random_state=42
    )

    x_train = x_train.drop(columns=["latency"], axis=1)
    x_test = x_test.drop(columns=["latency"], axis=1)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_norm = scaler.transform(x_train)
    x_test_norm = scaler.transform(x_test)

    # Use both train and test for symbolic regression (more data = better fit)
    x_all = np.vstack([x_train_norm, x_test_norm])
    feature_names = list(x_train.columns)

    # NN predictions (probabilities) as targets
    y_nn = model.predict(x_all, verbose=0).flatten()

    return x_all, y_nn, scaler, feature_names


def _patch_gplearn_sklearn_compat():
    """Patch BaseEstimator for gplearn compatibility with sklearn >= 1.2."""
    from sklearn.base import BaseEstimator
    from sklearn.utils.validation import check_X_y, check_array

    if not hasattr(BaseEstimator, "_validate_data"):

        def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
            if "y_numeric" in check_params:
                check_params = {k: v for k, v in check_params.items() if k != "y_numeric"}
            if y is None:
                out = check_array(X, **check_params)
            else:
                out = check_X_y(X, y, **check_params)
            if reset:
                X_val = out[0] if isinstance(out, tuple) else out
                self.n_features_in_ = X_val.shape[1] if len(X_val.shape) > 1 else 1
            return out

        BaseEstimator._validate_data = _validate_data


def run_symbolic_regression(
    X: np.ndarray,
    y_nn_proba: np.ndarray,
    feature_names: list[str],
    max_samples: int = 10000,
    sample_fraction: float | None = None,
    population_size: int = 1000,
    generations: int = 50,
    verbose: int = 0,
):
    """
    Run gplearn SymbolicClassifier to find expression predicting 0 or 1,
    mimicking the NN's binary decisions. Targets are NN outputs thresholded at 0.5.
    Returns (estimator, expression_str, metrics_dict).
    """
    try:
        from gplearn.genetic import SymbolicClassifier
    except ImportError:
        raise ImportError(
            "gplearn is required. Install with: pip install gplearn"
        ) from None

    _patch_gplearn_sklearn_compat()

    # Target: NN binary decisions (0 or 1)
    y_binary = (y_nn_proba > 0.5).astype(int)

    n_samples = X.shape[0]
    effective_max = int(n_samples * sample_fraction) if sample_fraction is not None else max_samples
    effective_max = max(1, min(effective_max, n_samples))
    if n_samples > effective_max:
        # Stratified sampling so both classes appear in training
        _, X_sub, _, y_sub = train_test_split(
            X, y_binary, train_size=effective_max, stratify=y_binary, random_state=42
        )
    else:
        X_sub = X
        y_sub = y_binary

    est = SymbolicClassifier(
        population_size=population_size,
        generations=generations,
        stopping_criteria=1e-6,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=verbose,
        parsimony_coefficient=0.001,
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"),
        init_depth=(2, 8),
        transformer="sigmoid",
    )

    est.fit(X_sub, y_sub)
    y_pred = est.predict(X)

    # Metrics: accuracy and agreement with NN binary decisions
    accuracy = accuracy_score(y_binary, y_pred)
    agreement = np.mean(y_binary == y_pred)
    mse = mean_squared_error(y_nn_proba, np.clip(est.predict_proba(X)[:, 1], 0, 1))
    r2 = r2_score(y_nn_proba, np.clip(est.predict_proba(X)[:, 1], 0, 1))
    y_proba_pred = est.predict_proba(X)[:, 1]
    corr = np.corrcoef(y_nn_proba, y_proba_pred)[0, 1] if len(y_nn_proba) > 1 else 0

    # Build human-readable expression with feature names (replace X11 before X1 to avoid partial matches)
    expr_str = str(est._program)
    for i in range(len(feature_names) - 1, -1, -1):
        expr_str = expr_str.replace(f"X{i}", feature_names[i])

    metrics = {
        "accuracy_vs_nn": float(accuracy),
        "agreement_with_nn": float(agreement),
        "mse": float(mse),
        "r2": float(r2),
        "correlation": float(corr) if not np.isnan(corr) else 0,
        "n_samples": int(n_samples),
        "n_samples_used": int(X_sub.shape[0]),
    }

    return est, expr_str, metrics


def save_results(output_dir: Path, expr_str: str, metrics: dict, feature_names: list[str], mldrive_name: str):
    """Save symbolic regression results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    expr_path = output_dir / "symbolic_regression_expression.txt"
    expr_path.write_text(expr_str, encoding="utf-8")

    metrics_path = output_dir / "symbolic_regression_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_lines = [
        "# Symbolic Regression Report",
        "",
        f"**Model**: {mldrive_name}/nnK",
        "",
        "## Discovered Expression",
        "",
        "Output is 0 or 1 (binary classification). Expression output is passed through sigmoid and thresholded.",
        "",
        "```",
        expr_str,
        "```",
        "",
        "## Metrics (vs NN binary decisions)",
        "",
        f"- Accuracy (agreement with NN): {metrics['accuracy_vs_nn']:.4f}",
        f"- R² (vs NN probabilities): {metrics['r2']:.4f}",
        f"- MSE (vs NN probabilities): {metrics['mse']:.6f}",
        f"- Correlation (vs NN probabilities): {metrics['correlation']:.4f}",
        f"- Samples used: {metrics['n_samples_used']} / {metrics['n_samples']}",
        "",
        "## Input Features",
        "",
    ]
    for i, name in enumerate(feature_names):
        report_lines.append(f"- X{i} = {name}")
    report_lines.append("")

    report_path = output_dir / "symbolic_regression_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return expr_path, metrics_path, report_path


def process_one_model(
    model_path: Path,
    dataset_path: Path,
    mldrive_name: str,
    train_eval_split: str = "50_50",
    max_samples: int = 10000,
    sample_fraction: float | None = None,
    population_size: int = 1000,
    generations: int = 50,
    verbose: int = 0,
):
    """Process a single model: load, run symbolic regression, save results."""
    nnk_dir = model_path.parent
    print(f"  Loading {mldrive_name}...")

    X, y_nn, scaler, feature_names = load_data_and_model(
        model_path, dataset_path, train_eval_split
    )
    print(f"  Data: {X.shape[0]} samples, {len(feature_names)} features")

    print(f"  Running symbolic regression...")
    est, expr_str, metrics = run_symbolic_regression(
        X, y_nn, feature_names,
        max_samples=max_samples,
        sample_fraction=sample_fraction,
        population_size=population_size,
        generations=generations,
        verbose=verbose,
    )

    paths = save_results(nnk_dir, expr_str, metrics, feature_names, mldrive_name)
    print(f"  Accuracy={metrics['accuracy_vs_nn']:.4f}  R²={metrics['r2']:.4f}  corr={metrics['correlation']:.4f}")
    print(f"  Expression: {expr_str[:80]}...")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Symbolic regression to mimic FlashNet nnK neural networks"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="Root data directory (default: integration/client-level/data)",
    )
    parser.add_argument(
        "-model",
        help="Path to model.keras (single-model mode)",
    )
    parser.add_argument(
        "-dataset",
        help="Path to mldrive*.csv (required with -model)",
    )
    parser.add_argument(
        "-train_eval_split",
        default="50_50",
        help="Train/eval split ratio (default: 50_50)",
    )
    parser.add_argument(
        "-max_samples",
        type=int,
        default=10000,
        help="Max samples for symbolic regression (default: 10000)",
    )
    parser.add_argument(
        "-sample_fraction",
        type=float,
        default=None,
        help="Use this fraction of samples (e.g. 0.1 for 10%%) instead of max_samples",
    )
    parser.add_argument(
        "-population_size",
        type=int,
        default=1000,
        help="GP population size (default: 1000)",
    )
    parser.add_argument(
        "-generations",
        type=int,
        default=50,
        help="GP generations (default: 50)",
    )
    parser.add_argument(
        "-verbose",
        type=int,
        default=0,
        choices=[0, 1],
        help="Verbosity: 0=quiet, 1=print GP progress per generation",
    )
    args = parser.parse_args()

    if args.model and args.dataset:
        model_path = Path(args.model).resolve()
        dataset_path = Path(args.dataset).resolve()
        if not model_path.exists():
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)
        if not dataset_path.exists():
            print(f"ERROR: Dataset not found: {dataset_path}")
            sys.exit(1)
        mldrive_name = dataset_path.stem
        process_one_model(
            model_path, dataset_path, mldrive_name,
            train_eval_split=args.train_eval_split,
            max_samples=args.max_samples,
            sample_fraction=args.sample_fraction,
            population_size=args.population_size,
            generations=args.generations,
            verbose=args.verbose,
        )
        return

    data_dir = Path(args.data_dir or (SCRIPT_DIR / ".." / ".." / ".." / "data")).resolve()
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    model_dirs = find_model_dirs(data_dir)
    if not model_dirs:
        print("No flashnet/nnK model.keras files found.")
        sys.exit(0)

    print(f"Found {len(model_dirs)} model(s) under {data_dir}\n")

    for model_path, dataset_path, mldrive_name in model_dirs:
        print(f"Processing: {model_path.relative_to(data_dir)}")
        try:
            process_one_model(
                model_path, dataset_path, mldrive_name,
                train_eval_split=args.train_eval_split,
                max_samples=args.max_samples,
                sample_fraction=args.sample_fraction,
                population_size=args.population_size,
                generations=args.generations,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("Done.")


if __name__ == "__main__":
    main()
