#!/usr/bin/env python3
"""
Local LLM Agent for Activation Maximization Insights

Analyzes activation maximization CSV data and generates text insight files for each neuron.
For each neuron, reports:
- Which input features are relatively higher when the neuron is activated
- Which other activations are high when the current neuron is maximized
- Separate analysis for each combination of: learning rate, other-neuron regularization,
  iterations, initialization, and input feature 0 value (f00 vs f01)

Requires: ollama (pip install ollama) and Ollama running locally with a model (e.g., ollama run llama3.2)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# Default feature names (FlashNet feat_v6: queue_len, io_type, size, prev_*, etc.)
DEFAULT_FEATURE_NAMES = [
    "queue_len", "io_type", "size",
    "prev_queue_len_1", "prev_queue_len_2", "prev_queue_len_3",
    "prev_latency_1", "prev_latency_2", "prev_latency_3",
    "prev_throughput_1", "prev_throughput_2", "prev_throughput_3",
]


def parse_activation_filename(filename: str) -> dict | None:
    """
    Parse activation maximization filename to extract parameters.
    Example: layer_0_neuron_37_loss_history_onr1.0_iter500_lr1.0_initrandom_f01.csv
    """
    # Pattern: layer_{layer}_neuron_{neuron}_{type}_onr{onr}_iter{iter}_lr{lr}_init{init}_f0{f}.csv
    pattern = (
        r"layer_(\d+)_neuron_(\d+)_"
        r"(maximizing_input|loss_history|all_activations)_"
        r"onr([\d.]+)_iter(\d+)_lr([\d.]+)_init(random|means)_f0(\d)\.csv"
    )
    m = re.match(pattern, filename)
    if not m:
        return None
    return {
        "layer": int(m.group(1)),
        "neuron": int(m.group(2)),
        "file_type": m.group(3),
        "onr": float(m.group(4)),
        "iter": int(m.group(5)),
        "lr": float(m.group(6)),
        "init": m.group(7),
        "f0": int(m.group(8)),
    }


def discover_activation_files(activation_dir: str) -> dict:
    """
    Discover all activation maximization CSV files and group by (layer, neuron).
    Returns: {(layer, neuron): {config_key: {maximizing_input_path, loss_history_path, all_activations_path}}}
    """
    activation_dir = Path(activation_dir)
    if not activation_dir.exists():
        return {}

    grouped = defaultdict(dict)

    for f in activation_dir.glob("*.csv"):
        parsed = parse_activation_filename(f.name)
        if parsed is None:
            continue
        layer, neuron = parsed["layer"], parsed["neuron"]
        config_key = (parsed["onr"], parsed["iter"], parsed["lr"], parsed["init"], parsed["f0"])

        if config_key not in grouped[(layer, neuron)]:
            grouped[(layer, neuron)][config_key] = {}

        if parsed["file_type"] == "maximizing_input":
            grouped[(layer, neuron)][config_key]["maximizing_input"] = str(f)
        elif parsed["file_type"] == "loss_history":
            grouped[(layer, neuron)][config_key]["loss_history"] = str(f)
        elif parsed["file_type"] == "all_activations":
            grouped[(layer, neuron)][config_key]["all_activations"] = str(f)

    return dict(grouped)


def load_maximizing_input(path: str) -> np.ndarray | None:
    """Load maximizing input CSV (single row of floats)."""
    try:
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 1:
            return data
        return data[0]
    except Exception:
        return None


def load_all_activations(path: str) -> pd.DataFrame | None:
    """Load all-activations CSV."""
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def analyze_input_features(
    maximizing_input: np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 5,
    threshold_percentile: float = 75,
) -> dict:
    """
    Identify which input features are relatively higher when the neuron is activated.
    Returns dict with top features by value and features above percentile.
    """
    n_features = len(maximizing_input)
    names = feature_names or [f"feature_{i}" for i in range(n_features)]
    if len(names) < n_features:
        names = names + [f"feature_{i}" for i in range(len(names), n_features)]
    names = names[:n_features]

    # Sort by absolute value (we care about magnitude for activation)
    indices = np.argsort(np.abs(maximizing_input))[::-1]
    top_indices = indices[:top_k]

    percentile_val = np.percentile(np.abs(maximizing_input), threshold_percentile)
    above_threshold = [
        (names[i], float(maximizing_input[i]))
        for i in range(n_features)
        if np.abs(maximizing_input[i]) >= percentile_val
    ]
    above_threshold.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "top_features_by_magnitude": [
            (names[i], float(maximizing_input[i])) for i in top_indices
        ],
        "features_above_75th_percentile": above_threshold[:top_k],
    }


def analyze_co_activations(
    activations_df: pd.DataFrame,
    target_layer: int,
    target_neuron: int,
    top_k: int = 5,
    exclude_same_layer: bool = False,
    same_or_earlier_layers_only: bool = True,
    positive_only: bool = True,
) -> dict:
    """
    Identify which other neurons have high activation when the target neuron is maximized.
    By default, only includes neurons in the same layer or earlier layers, with positive activations.
    """
    if activations_df is None or activations_df.empty:
        return {"high_activations": [], "target_activation": None}

    target_row = activations_df[
        (activations_df["layer_idx"] == target_layer)
        & (activations_df["neuron_idx"] == target_neuron)
    ]
    target_activation = (
        float(target_row["activation_value"].iloc[0])
        if len(target_row) > 0
        else None
    )

    # Exclude target neuron
    mask = (activations_df["layer_idx"] != target_layer) | (
        activations_df["neuron_idx"] != target_neuron
    )
    if exclude_same_layer:
        mask = mask & (activations_df["layer_idx"] != target_layer)
    if same_or_earlier_layers_only:
        mask = mask & (activations_df["layer_idx"] <= target_layer)
    if positive_only:
        mask = mask & (activations_df["activation_value"] > 0)
    others = activations_df[mask].copy()
    others = others.sort_values("activation_value", ascending=False).head(top_k)

    high_activations = [
        {
            "layer": int(row["layer_idx"]),
            "neuron": int(row["neuron_idx"]),
            "activation": float(row["activation_value"]),
        }
        for _, row in others.iterrows()
    ]

    return {
        "high_activations": high_activations,
        "target_activation": target_activation,
    }


def run_analysis_for_neuron(
    layer: int,
    neuron: int,
    configs: dict,
    feature_names: list[str] | None = None,
    top_k_features: int = 5,
    top_k_activations: int = 5,
) -> dict:
    """
    Run full analysis for one neuron across all configs.
    Returns structured data for LLM or template.
    """
    results = {
        "layer": layer,
        "neuron": neuron,
        "configs": {},
    }

    for config_key, paths in configs.items():
        onr, iter_val, lr, init, f0 = config_key
        config_label = f"onr={onr}, iter={iter_val}, lr={lr}, init={init}, f0={f0}"

        input_analysis = None
        activation_analysis = None

        if "maximizing_input" in paths:
            inp = load_maximizing_input(paths["maximizing_input"])
            if inp is not None:
                input_analysis = analyze_input_features(
                    inp,
                    feature_names=feature_names,
                    top_k=top_k_features,
                )

        if "all_activations" in paths:
            act_df = load_all_activations(paths["all_activations"])
            if act_df is not None:
                activation_analysis = analyze_co_activations(
                    act_df,
                    target_layer=layer,
                    target_neuron=neuron,
                    top_k=top_k_activations,
                )

        results["configs"][config_label] = {
            "input_features": input_analysis,
            "co_activations": activation_analysis,
        }

    return results


def format_analysis_for_llm(analysis: dict) -> str:
    """Format analysis results as text for LLM prompt."""
    lines = [
        f"# Neuron Analysis: Layer {analysis['layer']}, Neuron {analysis['neuron']}",
        "",
    ]
    for config_label, data in analysis["configs"].items():
        lines.append(f"## Config: {config_label}")
        lines.append("")
        if data.get("input_features"):
            inp = data["input_features"]
            lines.append("### Input features relatively higher when neuron is activated:")
            for name, val in inp.get("top_features_by_magnitude", [])[:5]:
                lines.append(f"  - {name}: {val:.4f}")
            lines.append("")
        if data.get("co_activations"):
            co = data["co_activations"]
            lines.append("### Other neurons with high activation when this neuron is maximized:")
            for item in co.get("high_activations", [])[:5]:
                lines.append(
                    f"  - Layer {item['layer']} Neuron {item['neuron']}: {item['activation']:.2f}"
                )
            if co.get("target_activation") is not None:
                lines.append(f"  (Target neuron activation: {co['target_activation']:.2f})")
            lines.append("")
        lines.append("")
    return "\n".join(lines)


def generate_insight_with_ollama(
    analysis_text: str,
    model: str = "llama3.2",
    system_prompt: str | None = None,
) -> str:
    """
    Use Ollama to generate natural language insights from the analysis.
    Falls back to returning the raw analysis if Ollama is unavailable.
    """
    try:
        import ollama
    except ImportError:
        return (
            "Ollama not installed. Run: pip install ollama\n\n"
            + "Raw analysis:\n\n"
            + analysis_text
        )

    system = system_prompt or (
        "You are an expert in neural network interpretability. "
        "Given activation maximization analysis data, write a concise, clear summary. "
        "For each configuration (onr, iter, lr, init, f0), explain which input features "
        "drive the neuron's activation and which other neurons tend to co-activate. "
        "Use plain language suitable for a technical report. Be specific with numbers."
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": analysis_text},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        return (
            f"Ollama error ({e}). Ensure Ollama is running (ollama serve) and model exists (ollama run {model}).\n\n"
            "Raw analysis:\n\n"
            + analysis_text
        )


def generate_insight_template(analysis: dict, feature_names: list[str] | None) -> str:
    """
    Generate insight text using a template (no LLM required).
    """
    lines = [
        "=" * 70,
        f"ACTIVATION MAXIMIZATION INSIGHTS: Layer {analysis['layer']}, Neuron {analysis['neuron']}",
        "=" * 70,
        "",
        "This report summarizes which input features and co-activating neurons",
        "are associated with maximal activation of this neuron, for each",
        "combination of hyperparameters (onr, iter, lr, init) and input feature 0 value (f0).",
        "",
    ]

    # Group by f0 for clearer structure
    by_f0 = defaultdict(dict)
    for config_label, data in analysis["configs"].items():
        # Extract f0 from label (last part)
        f0_match = re.search(r"f0=(\d)", config_label)
        f0 = int(f0_match.group(1)) if f0_match else 0
        by_f0[f0][config_label] = data

    for f0 in sorted(by_f0.keys()):
        lines.append("")
        lines.append("-" * 50)
        lines.append(f"INPUT FEATURE 0 = {f0} (f0{f0})")
        lines.append("-" * 50)

        for config_label, data in by_f0[f0].items():
            lines.append("")
            lines.append(f"  Config: {config_label}")
            lines.append("")

            if data.get("input_features"):
                inp = data["input_features"]
                lines.append("  Input features relatively higher when neuron is activated:")
                for name, val in inp.get("top_features_by_magnitude", [])[:5]:
                    lines.append(f"    - {name}: {val:.4f}")
                lines.append("")

            if data.get("co_activations"):
                co = data["co_activations"]
                lines.append("  Other neurons with high activation when this neuron is maximized:")
                for item in co.get("high_activations", [])[:5]:
                    lines.append(
                        f"    - Layer {item['layer']} Neuron {item['neuron']}: {item['activation']:.2f}"
                    )
                if co.get("target_activation") is not None:
                    lines.append(f"    (This neuron's activation: {co['target_activation']:.2f})")
                lines.append("")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def process_activation_directory(
    activation_dir: str,
    output_dir: str | None = None,
    feature_names: list[str] | None = None,
    use_llm: bool = True,
    llm_model: str = "llama3.2",
    top_k_features: int = 5,
    top_k_activations: int = 5,
) -> list[str]:
    """
    Process all neurons in an activation maximization directory.
    Writes one insight file per neuron.
    Returns list of output file paths.
    """
    activation_dir = Path(activation_dir)
    output_dir = Path(output_dir or activation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = discover_activation_files(str(activation_dir))
    if not grouped:
        print(f"No activation maximization files found in {activation_dir}")
        return []

    feature_names = feature_names or DEFAULT_FEATURE_NAMES
    output_files = []

    for (layer, neuron), configs in sorted(grouped.items()):
        # Need at least one config with maximizing_input
        has_input = any("maximizing_input" in p for p in configs.values())
        if not has_input:
            continue

        print(f"  Processing Layer {layer} Neuron {neuron}...", end=" ", flush=True)
        analysis = run_analysis_for_neuron(
            layer, neuron, configs,
            feature_names=feature_names,
            top_k_features=top_k_features,
            top_k_activations=top_k_activations,
        )

        if use_llm:
            analysis_text = format_analysis_for_llm(analysis)
            insight = generate_insight_with_ollama(analysis_text, model=llm_model)
        else:
            insight = generate_insight_template(analysis, feature_names)

        out_path = output_dir / f"layer_{layer}_neuron_{neuron}_insights.txt"
        out_path.write_text(insight, encoding="utf-8")
        output_files.append(str(out_path))
        print(f" -> {out_path.name}")

    return output_files


def find_activation_dirs(data_root: str) -> list[str]:
    """Find all activation_maximization directories under data_root."""
    data_root = Path(data_root)
    dirs = list(data_root.rglob("activation_maximization"))
    return [str(d) for d in dirs if d.is_dir()]


def main():
    parser = argparse.ArgumentParser(
        description="Generate activation maximization insight files using a local LLM"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to activation_maximization directory or data root to search",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory for insight files (default: same as input)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use template-based output instead of LLM (no Ollama required)",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model name (default: llama3.2)",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=5,
        help="Number of top input features to report (default: 5)",
    )
    parser.add_argument(
        "--top-activations",
        type=int,
        default=5,
        help="Number of top co-activating neurons to report (default: 5)",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Search path recursively for activation_maximization directories",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        help="Feature names (default: FlashNet feat_v6 names)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: Path not found: {path}")
        sys.exit(1)

    if args.search:
        activation_dirs = find_activation_dirs(str(path))
        if not activation_dirs:
            print(f"No activation_maximization directories found under {path}")
            sys.exit(1)
        print(f"Found {len(activation_dirs)} activation_maximization directory(ies)")
    else:
        if (path / "activation_maximization").exists():
            activation_dirs = [str(path / "activation_maximization")]
        elif path.name == "activation_maximization":
            activation_dirs = [str(path)]
        else:
            activation_dirs = [str(path)]

    feature_names = args.features
    total = 0
    for act_dir in activation_dirs:
        print(f"\nProcessing: {act_dir}")
        output_dir = args.output
        if output_dir is None and not args.search:
            output_dir = act_dir
        elif output_dir is None and args.search:
            output_dir = str(Path(act_dir).parent / "activation_insights")
        files = process_activation_directory(
            act_dir,
            output_dir=output_dir,
            feature_names=feature_names,
            use_llm=not args.no_llm,
            llm_model=args.model,
            top_k_features=args.top_features,
            top_k_activations=args.top_activations,
        )
        total += len(files)

    print(f"\nDone. Generated {total} insight file(s).")


if __name__ == "__main__":
    main()
