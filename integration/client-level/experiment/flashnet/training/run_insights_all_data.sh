#!/bin/bash
#
# Run activation insights analysis for all activation_maximization directories
# under the data directory. Outputs are stored in data/insights/ with a structure
# mirroring the data directory.
#
# Usage: ./run_insights_all_data.sh [data_directory]
#
# Example:
#   ./run_insights_all_data.sh
#   ./run_insights_all_data.sh /path/to/data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-$SCRIPT_DIR/../../../data}"

python3 "$SCRIPT_DIR/run_insights_all_data.py" "$DATA_DIR" --no-llm
