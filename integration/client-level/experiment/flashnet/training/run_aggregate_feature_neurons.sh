#!/bin/bash
#
# Aggregate insight files into per-feature neuron lists.
# Reads all insight files under data/insights/ and produces one CSV per input feature
# with neurons that are maximized when that feature is in the top activators.
#
# For feature 0 (queue_len): produces queue_len_f0_0.csv and queue_len_f0_1.csv
#
# Usage: ./run_aggregate_feature_neurons.sh [insights_directory]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSIGHTS_ROOT="${1:-$SCRIPT_DIR/../../../data/insights}"

python3 "$SCRIPT_DIR/aggregate_feature_neurons.py" "$INSIGHTS_ROOT"
