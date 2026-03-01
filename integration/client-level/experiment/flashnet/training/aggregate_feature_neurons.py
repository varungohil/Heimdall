#!/usr/bin/env python3
"""
Read all insight files and produce an output file for each input feature.
Each output file contains a column of all neurons that are maximized when that
input feature appears in the top features.

For feature 0 (queue_len), produces two output files: one when f0=0 and one when f0=1.

Usage: python aggregate_feature_neurons.py [insights_root] [-o output_dir]
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


# Feature 0 name (used for f0 split)
FEATURE_0_NAME = "queue_len"


def parse_insight_filename(path: Path) -> tuple[int, int] | None:
    """Extract (layer, neuron) from filename like layer_0_neuron_28_insights.txt"""
    m = re.match(r"layer_(\d+)_neuron_(\d+)_insights\.txt", path.name)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def parse_insight_file(path: Path) -> list[tuple[int, int, int | None, list[str]]]:
    """
    Parse an insight file and yield (layer, neuron, f0, features) for each config block.
    f0 is 0 or 1 from section header, or None if not in a section.
    features is list of feature names from "Input features relatively higher".
    """
    parsed = parse_insight_filename(path)
    if parsed is None:
        return []
    layer, neuron = parsed

    content = path.read_text(encoding="utf-8")
    results = []

    # Split by section headers: INPUT FEATURE 0 = 0 (f00) or INPUT FEATURE 0 = 1 (f01)
    section_pattern = re.compile(
        r"INPUT FEATURE 0 = ([01]) \(f0[01]\)\s*\n(.*?)(?=INPUT FEATURE 0 =|\Z)",
        re.DOTALL,
    )
    feature_line_pattern = re.compile(r"^\s*-\s+([^:]+):\s*[\d.-]+", re.MULTILINE)

    for section_match in section_pattern.finditer(content):
        f0 = int(section_match.group(1))
        section_text = section_match.group(2)

        # Find each config block (starts with "Config:") and its "Input features" list
        config_blocks = re.split(r"\n\s+Config:", section_text)
        for block in config_blocks:
            if "Input features relatively higher" not in block:
                continue
            # Extract the features list (between "Input features..." and "Other neurons...")
            features_section = re.search(
                r"Input features relatively higher when neuron is activated:\s*\n(.*?)(?=Other neurons|$)",
                block,
                re.DOTALL,
            )
            if not features_section:
                continue
            features_text = features_section.group(1)
            feature_names = []
            for line in features_text.strip().split("\n"):
                m = feature_line_pattern.match(line)
                if m:
                    feature_names.append(m.group(1).strip())
            if feature_names:
                results.append((layer, neuron, f0, feature_names))

    return results


def collect_feature_neurons_for_dir(
    insight_paths: list[Path],
) -> dict[str, set[tuple[int, int]]]:
    """
    Build mapping: feature_key -> set of (layer, neuron) from a list of insight files.
    A neuron is included only if the feature appears in ALL config blocks for that neuron.
    For queue_len, uses only config blocks with the matching f0 value.
    """
    # (layer, neuron) -> list of (f0, features) for each config block
    neuron_configs: dict[tuple[int, int], list[tuple[int, list[str]]]] = defaultdict(list)

    for path in insight_paths:
        parsed = parse_insight_filename(path)
        if parsed is None:
            continue
        layer, neuron = parsed
        for _, _, f0, features in parse_insight_file(path):
            neuron_configs[(layer, neuron)].append((f0, features))

    feature_neurons: dict[str, set[tuple[int, int]]] = defaultdict(set)

    for (layer, neuron), configs in neuron_configs.items():
        if not configs:
            continue

        # Other features: must appear in ALL config blocks
        all_feature_sets = [set(feats) for _, feats in configs]
        features_in_all = all_feature_sets[0].copy()
        for s in all_feature_sets[1:]:
            features_in_all &= s

        # queue_len_f0_0: must appear in all config blocks where f0=0
        f0_0_configs = [feats for f0, feats in configs if f0 == 0]
        f0_0_features: set[str] = set()
        if f0_0_configs:
            f0_0_features = set(f0_0_configs[0])
            for feats in f0_0_configs[1:]:
                f0_0_features &= set(feats)

        # queue_len_f0_1: must appear in all config blocks where f0=1
        f0_1_configs = [feats for f0, feats in configs if f0 == 1]
        f0_1_features: set[str] = set()
        if f0_1_configs:
            f0_1_features = set(f0_1_configs[0])
            for feats in f0_1_configs[1:]:
                f0_1_features &= set(feats)

        for feat in features_in_all:
            if feat != FEATURE_0_NAME:
                feature_neurons[feat].add((layer, neuron))
        if FEATURE_0_NAME in f0_0_features:
            feature_neurons[f"{FEATURE_0_NAME}_f0_0"].add((layer, neuron))
        if FEATURE_0_NAME in f0_1_features:
            feature_neurons[f"{FEATURE_0_NAME}_f0_1"].add((layer, neuron))

    return dict(feature_neurons)


def group_insight_files_by_dir(insights_root: Path) -> dict[Path, list[Path]]:
    """
    Group insight files by their parent activation_insights directory.
    Returns: {activation_insights_dir: [list of insight file paths]}
    """
    groups: dict[Path, list[Path]] = defaultdict(list)
    for insight_path in insights_root.rglob("layer_*_neuron_*_insights.txt"):
        parent = insight_path.parent
        groups[parent].append(insight_path)
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate insight files into per-feature neuron lists"
    )
    parser.add_argument(
        "insights_root",
        nargs="?",
        default=None,
        help="Root directory containing insight files (default: data/insights)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output directory for feature files (default: insights_root/feature_summaries)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_insights = script_dir / ".." / ".." / ".." / "data" / "insights"
    insights_root = Path(args.insights_root or default_insights).resolve()

    if not insights_root.exists():
        print(f"ERROR: Insights root not found: {insights_root}")
        sys.exit(1)

    feature_summaries_root = Path(
        args.output or (insights_root / "feature_summaries")
    ).resolve()

    print(f"Scanning: {insights_root}")
    print(f"Output:   {feature_summaries_root}")
    print()

    groups = group_insight_files_by_dir(insights_root)
    if not groups:
        print("No insight files found.")
        sys.exit(0)

    total_files = 0
    for activation_insights_dir, insight_paths in sorted(groups.items()):
        try:
            rel_path = activation_insights_dir.relative_to(insights_root)
        except ValueError:
            rel_path = activation_insights_dir.name
        out_dir = feature_summaries_root / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)

        feature_neurons = collect_feature_neurons_for_dir(insight_paths)
        if not feature_neurons:
            continue

        print(f"  {rel_path}")
        for feature_key in sorted(feature_neurons.keys()):
            neurons = sorted(feature_neurons[feature_key])
            out_path = out_dir / f"{feature_key}.csv"
            with open(out_path, "w") as f:
                f.write("layer,neuron\n")
                for layer, neuron in neurons:
                    f.write(f"{layer},{neuron}\n")
            total_files += 1
            print(f"    {feature_key}: {len(neurons)} neurons -> {out_path.name}")

    print(f"\nDone. Wrote {total_files} feature files.")


if __name__ == "__main__":
    main()
