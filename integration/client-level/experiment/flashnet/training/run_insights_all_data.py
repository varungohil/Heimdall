#!/usr/bin/env python3
"""
Run activation insights analysis for all activation_maximization directories
under the data directory. Outputs are stored in data/insights/ with a structure
mirroring the data directory.

Example structure:
  data/
    alibaba.../.../mldrive0/nnK/activation_maximization/
  data/insights/
    alibaba.../.../mldrive0/nnK/activation_insights/

Usage: python run_insights_all_data.py [data_directory]
"""

import argparse
import sys
from pathlib import Path

# Add script directory for imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from activation_insights_agent import (
    find_activation_dirs,
    process_activation_directory,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run activation insights for all activation_maximization dirs under data"
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=str(SCRIPT_DIR / ".." / ".." / ".." / "data"),
        help="Root data directory (default: integration/client-level/data)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use template output instead of LLM",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model (default: llama3.2)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    insights_root = data_dir / "insights"
    insights_root.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {data_dir}")
    print(f"Insights root: {insights_root}")
    print()

    activation_dirs = find_activation_dirs(str(data_dir))
    if not activation_dirs:
        print("No activation_maximization directories found.")
        sys.exit(0)

    print(f"Found {len(activation_dirs)} activation_maximization directory(ies)\n")

    total_files = 0
    for act_dir in sorted(activation_dirs):
        act_path = Path(act_dir)
        # Compute relative path from data_dir to activation_maximization
        try:
            rel = act_path.relative_to(data_dir)
        except ValueError:
            # act_dir might not be under data_dir if they passed a different root
            rel = act_path.name
        # Replace activation_maximization with activation_insights in the path
        rel_parts = list(rel.parts)
        if "activation_maximization" in rel_parts:
            idx = rel_parts.index("activation_maximization")
            rel_parts[idx] = "activation_insights"
        else:
            rel_parts.append("activation_insights")
        output_dir = insights_root / Path(*rel_parts)

        print(f"Processing: {act_dir}")
        print(f"  -> Output: {output_dir}")
        files = process_activation_directory(
            act_dir,
            output_dir=str(output_dir),
            use_llm=not args.no_llm,
            llm_model=args.model,
        )
        total_files += len(files)
        print()

    print("=" * 50)
    print(f"Done. Generated {total_files} insight file(s)")

    if total_files > 0:
        print(f"Output location: {insights_root}")


if __name__ == "__main__":
    main()
