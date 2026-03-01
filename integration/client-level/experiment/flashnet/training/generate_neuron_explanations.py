#!/usr/bin/env python3
"""
Generate natural language explanations for each neuron in a FlashNet model.
Uses: model weights, activation insights, and feature summaries.

Output: One .txt file per neuron with a human-readable explanation.

Usage: python generate_neuron_explanations.py <data_folder>
Example: python generate_neuron_explanations.py ../../data/alibaba.per_3mins.iops_p10.alibaba_50.97/alibaba.per_3mins.iops_p100.alibaba_9086.35
"""

import argparse
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

# Feature names for interpretation
FEATURE_NAMES = [
    "queue_len", "io_type", "size",
    "prev_queue_len_1", "prev_queue_len_2", "prev_queue_len_3",
    "prev_latency_1", "prev_latency_2", "prev_latency_3",
    "prev_throughput_1", "prev_throughput_2", "prev_throughput_3",
]

FEATURE_DESCRIPTIONS = {
    "queue_len": "current number of IOs waiting in the queue",
    "io_type": "whether the IO is a read or write",
    "size": "size of the IO in bytes",
    "prev_queue_len_1": "queue length from 1 IO ago",
    "prev_queue_len_2": "queue length from 2 IOs ago",
    "prev_queue_len_3": "queue length from 3 IOs ago",
    "prev_latency_1": "how slow the most recent IO was",
    "prev_latency_2": "how slow the 2nd-most-recent IO was",
    "prev_latency_3": "how slow the 3rd-most-recent IO was",
    "prev_throughput_1": "how fast the most recent IO completed",
    "prev_throughput_2": "throughput from 2 IOs ago",
    "prev_throughput_3": "throughput from 3 IOs ago",
}

# For "fires when" phrases: (high_phrase, low_phrase) - positive weight -> fires when high, negative -> fires when low
FEATURE_FIRES_WHEN = {
    "queue_len": ("high queue length", "low queue length"),
    "io_type": ("write-heavy", "read-heavy"),
    "size": ("large IO size", "small IO size"),
    "prev_queue_len_1": ("recently busy queue", "recently idle queue"),
    "prev_queue_len_2": ("queue was busy 2 IOs ago", "queue was idle 2 IOs ago"),
    "prev_queue_len_3": ("queue was busy 3 IOs ago", "queue was idle 3 IOs ago"),
    "prev_latency_1": ("recently high latency", "recently low latency"),
    "prev_latency_2": ("high latency 2 IOs ago", "low latency 2 IOs ago"),
    "prev_latency_3": ("high latency 3 IOs ago", "low latency 3 IOs ago"),
    "prev_throughput_1": ("recently high throughput", "recently low throughput"),
    "prev_throughput_2": ("high throughput 2 IOs ago", "low throughput 2 IOs ago"),
    "prev_throughput_3": ("high throughput 3 IOs ago", "low throughput 3 IOs ago"),
}


def load_model_weights(model_path: Path) -> dict:
    """Load Keras model and extract weight matrices for each Dense layer.
    Builds the architecture and loads weights from .keras (avoids deserialization issues)."""
    from tensorflow import keras
    from tensorflow.keras import layers

    # Build same architecture as nnK.py (avoids batch_input_shape deserialization errors)
    model = keras.Sequential([
        layers.Dense(128, input_dim=12, name="dense_1"),
        layers.Activation("relu", name="relu_1"),
        layers.Dense(16, name="dense_2"),
        layers.Activation("relu", name="relu_2"),
        layers.Dense(1, name="dense_3"),
        layers.Activation("sigmoid", name="sigmoid_out"),
    ])

    # .keras is a zip; extract model.weights.h5 and load
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

    weights = {}
    w0, b0 = model.layers[0].get_weights()
    weights["layer_0"] = {"W": w0, "b": b0}
    w2, b2 = model.layers[2].get_weights()
    weights["layer_2"] = {"W": w2, "b": b2}
    w4, b4 = model.layers[4].get_weights()
    weights["layer_4"] = {"W": w4, "b": b4}
    return weights


def get_top_inputs_for_neuron(weights: dict, layer: int, neuron: int, top_k: int = 5) -> list[tuple[str, float]]:
    """Get top input features (by weight magnitude) for a Layer 0 neuron."""
    W = weights["layer_0"]["W"]  # (12, 128), W[feat_idx, neuron_idx]
    col = W[:, neuron]
    indices = np.argsort(np.abs(col))[::-1][:top_k]
    return [(FEATURE_NAMES[i], float(col[i])) for i in indices]


def get_top_upstream_neurons(weights: dict, layer: int, neuron: int, top_k: int = 5) -> list[tuple[int, float]]:
    """Get top upstream neurons (by weight magnitude) for Layer 2 or 4."""
    if layer == 2:
        W = weights["layer_2"]["W"]  # (128, 16)
        col = W[:, neuron]
        indices = np.argsort(np.abs(col))[::-1][:top_k]
        return [(int(i), float(col[i])) for i in indices]
    elif layer == 4:
        W = weights["layer_4"]["W"]  # (16, 1)
        col = W[:, 0]
        indices = np.argsort(np.abs(col))[::-1][:top_k]
        return [(int(i), float(col[i])) for i in indices]
    return []


def parse_insight_file(path: Path) -> dict | None:
    """Parse insight file and extract key info: top features, co-activating neurons."""
    if not path.exists():
        return None

    content = path.read_text(encoding="utf-8")
    result = {"top_features": [], "co_activations": []}

    # Get first config block's top features as representative
    feat_match = re.search(
        r"Input features relatively higher when neuron is activated:\s*\n(.*?)(?=Other neurons|$)",
        content,
        re.DOTALL,
    )
    if feat_match:
        for line in feat_match.group(1).strip().split("\n"):
            m = re.match(r"\s*-\s+([^:]+):\s*([\d.-]+)", line)
            if m:
                result["top_features"].append((m.group(1).strip(), float(m.group(2))))

    # Get co-activating neurons
    co_match = re.search(
        r"Other neurons with high activation when this neuron is maximized:\s*\n(.*?)(?=\(This neuron's|$)",
        content,
        re.DOTALL,
    )
    if co_match:
        for line in co_match.group(1).strip().split("\n"):
            m = re.match(r"\s*-\s+Layer\s+(\d+)\s+Neuron\s+(\d+):\s*([\d.-]+)", line)
            if m:
                result["co_activations"].append(
                    (int(m.group(1)), int(m.group(2)), float(m.group(3)))
                )

    return result


def load_feature_summaries(feature_summaries_dir: Path) -> dict[str, set[tuple[int, int]]]:
    """Load feature summary CSVs: feature -> set of (layer, neuron)."""
    result = {}
    if not feature_summaries_dir.exists():
        return result

    for csv_path in feature_summaries_dir.glob("*.csv"):
        feat_name = csv_path.stem
        try:
            df = __import__("pandas").read_csv(csv_path)
            neurons = set(zip(df["layer"].astype(int), df["neuron"].astype(int)))
            result[feat_name] = neurons
        except Exception:
            pass
    return result


def neurons_in_feature(feature_summaries: dict, layer: int, neuron: int) -> list[str]:
    """Which features include this neuron in their summary?"""
    return [
        feat for feat, neurons in feature_summaries.items()
        if (layer, neuron) in neurons
    ]


def _base_feature(feat: str) -> str:
    """Get base feature name (strip _f0_0, _f0_1 etc)."""
    for suffix in ("_f0_0", "_f0_1", "_f0_2", "_f0_3"):
        if feat.endswith(suffix):
            return feat[: -len(suffix)]
    return feat


def build_fires_when_phrase(top_inputs: list[tuple[str, float]], top_k: int = 5) -> str:
    """Build 'fires when X, Y, and Z' from top input features and weights.
    Positive weight -> fires when feature is high; negative -> fires when low."""
    phrases = []
    for feat, w in top_inputs[:top_k]:
        base = _base_feature(feat)
        high_phrase, low_phrase = FEATURE_FIRES_WHEN.get(base, (f"high {base}", f"low {base}"))
        phrases.append(high_phrase if w > 0 else low_phrase)
    if not phrases:
        return "certain workload patterns"
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def build_fires_when_from_insight(insight: dict | None, top_k: int = 5) -> str | None:
    """Build 'fires when' phrase from activation maximization insight (top_features with values)."""
    if not insight or not insight.get("top_features"):
        return None
    phrases = []
    for feat, val in insight["top_features"][:top_k]:
        base = _base_feature(feat)
        high_phrase, low_phrase = FEATURE_FIRES_WHEN.get(base, (f"high {base}", f"low {base}"))
        phrases.append(high_phrase if val > 0 else low_phrase)
    if not phrases:
        return None
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def generate_explanation(
    layer: int,
    neuron: int,
    weights: dict,
    insight: dict | None,
    feature_summaries: dict,
) -> str:
    """Generate natural language explanation for a neuron."""
    lines = []
    lines.append(f"# Neuron Explanation: Layer {layer}, Neuron {neuron}")
    lines.append("")

    if layer == 0:
        top_inputs = get_top_inputs_for_neuron(weights, layer, neuron)
        fires_when = build_fires_when_phrase(top_inputs)
        lines.append("## What this neuron represents")
        lines.append("")
        lines.append(f"**This neuron fires when:** {fires_when}.")
        lines.append("")
        lines.append("(From model weights: strongest input influences)")
        for feat, w in top_inputs[:5]:
            base = _base_feature(feat)
            high_phrase, low_phrase = FEATURE_FIRES_WHEN.get(base, (f"high {base}", f"low {base}"))
            phrase = high_phrase if w > 0 else low_phrase
            lines.append(f"- {phrase} (weight={w:+.3f})")

    elif layer == 2:
        top_upstream = get_top_upstream_neurons(weights, layer, neuron)
        fires_when = build_fires_when_from_insight(insight) or "certain combinations of Layer 0 signals"
        lines.append("## What this neuron represents")
        lines.append("")
        lines.append(f"**This neuron fires when:** {fires_when}.")
        lines.append("")
        lines.append("It combines signals from these Layer 0 neurons:")
        for up_neuron, w in top_upstream[:5]:
            direction = "excites" if w > 0 else "inhibits"
            lines.append(f"- Layer 0 Neuron {up_neuron}: {direction} (weight={w:+.3f})")

    elif layer == 4:
        top_upstream = get_top_upstream_neurons(weights, layer, neuron)
        fires_when = build_fires_when_from_insight(insight) or "conditions favor rejection"
        lines.append("## What this neuron represents")
        lines.append("")
        lines.append("**This is the output neuron.** It fires (rejects the IO) when: " + fires_when + ".")
        lines.append("")
        lines.append("Strongest influences from Layer 2:")
        for up_neuron, w in top_upstream[:5]:
            direction = "REJECT" if w > 0 else "ACCEPT"
            lines.append(f"- Layer 2 Neuron {up_neuron}: pushes toward {direction} (weight={w:+.3f})")

    # Feature summary: which features consistently include this neuron
    features_with_neuron = neurons_in_feature(feature_summaries, layer, neuron)
    if features_with_neuron:
        lines.append("")
        lines.append("## Consistently associated features")
        lines.append("")
        lines.append("Across all hyperparameter settings, this neuron appears in the top activators for: "
                    + ", ".join(features_with_neuron[:10]))
        if len(features_with_neuron) > 10:
            lines.append(f" (and {len(features_with_neuron) - 10} more)")

    # Co-activating neurons from insights
    if insight and insight.get("co_activations"):
        lines.append("")
        lines.append("## Neurons that tend to fire together")
        lines.append("")
        lines.append("When this neuron is maximized, these other neurons also tend to be active:")
        for l, n, val in insight["co_activations"][:5]:
            lines.append(f"- Layer {l} Neuron {n} (activation={val:.2f})")

    lines.append("")
    lines.append("---")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate neuron explanation files")
    parser.add_argument("data_folder", help="Path to data folder (e.g. .../alibaba.per_3mins.iops_p100.alibaba_9086.35)")
    parser.add_argument("-o", "--output", help="Output directory (default: data_folder/neuron_explanations)")
    args = parser.parse_args()

    data_folder = Path(args.data_folder).resolve()
    if not data_folder.exists():
        print(f"ERROR: Data folder not found: {data_folder}")
        sys.exit(1)

    # Find model under data_folder
    models = [p for p in data_folder.rglob("model.keras") if "flashnet" in str(p) and "nnK" in str(p) and "mldrive0" in str(p)]
    if not models:
        models = list(data_folder.rglob("model.keras"))
    if not models:
        print("ERROR: No model.keras found under data folder")
        sys.exit(1)
    model_path = models[0]
    nnk_dir = model_path.parent

    # Insights: data/.../nnK -> data/insights/.../nnK/activation_insights
    data_root = None
    for parent in [data_folder] + list(data_folder.parents):
        if parent.name == "data":
            data_root = parent
            break
    if data_root is None:
        data_root = Path("/users/varuncg/Heimdall/integration/client-level/data")

    insights_dir = None
    try:
        rel = nnk_dir.relative_to(data_root)
        candidate = data_root / "insights" / rel / "activation_insights"
        if candidate.exists() and (candidate / "layer_0_neuron_0_insights.txt").exists():
            insights_dir = candidate
    except ValueError:
        pass
    if insights_dir is None:
        for p in (data_root / "insights").rglob("layer_0_neuron_0_insights.txt"):
            if "mldrive0" in str(p) and "nnK" in str(p) and str(nnk_dir) in str(p):
                insights_dir = p.parent
                break
        if insights_dir is None:
            for p in (data_root / "insights").rglob("layer_0_neuron_0_insights.txt"):
                if "mldrive0" in str(p) and "nnK" in str(p):
                    insights_dir = p.parent
                    break

    # Feature summaries: data/insights/feature_summaries/.../nnK/activation_insights
    feature_summaries_dir = None
    if insights_dir:
        try:
            rel = insights_dir.relative_to(data_root / "insights")
            candidate = data_root / "insights" / "feature_summaries" / rel
            if candidate.exists() and (candidate / "io_type.csv").exists():
                feature_summaries_dir = candidate
        except ValueError:
            pass
    if feature_summaries_dir is None and (data_root / "insights" / "feature_summaries").exists():
        for p in (data_root / "insights" / "feature_summaries").rglob("activation_insights"):
            if "mldrive0" in str(p) and "nnK" in str(p) and (p / "io_type.csv").exists():
                feature_summaries_dir = p
                break

    output_dir = Path(args.output or (data_folder / "neuron_explanations")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_path}")
    print(f"Insights: {insights_dir}")
    print(f"Feature summaries: {feature_summaries_dir}")
    print(f"Output: {output_dir}")
    print()

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    weights = load_model_weights(model_path)
    feature_summaries = load_feature_summaries(feature_summaries_dir) if feature_summaries_dir and feature_summaries_dir.exists() else {}

    neuron_counts = [(0, 128), (2, 16), (4, 1)]
    for layer, count in neuron_counts:
        for neuron in range(count):
            insight_path = insights_dir / f"layer_{layer}_neuron_{neuron}_insights.txt" if insights_dir and insights_dir.exists() else None
            insight = parse_insight_file(insight_path) if insight_path else None

            explanation = generate_explanation(layer, neuron, weights, insight, feature_summaries)
            out_path = output_dir / f"layer_{layer}_neuron_{neuron}_explanation.txt"
            out_path.write_text(explanation, encoding="utf-8")
            print(f"  Wrote {out_path.name}")

    print(f"\nDone. Generated {sum(c for _, c in neuron_counts)} explanation files in {output_dir}")


if __name__ == "__main__":
    main()
