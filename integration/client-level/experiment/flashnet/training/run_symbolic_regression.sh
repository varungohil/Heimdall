#!/bin/bash
# Run symbolic regression for all FlashNet nnK models under the data directory.
# Outputs: symbolic_regression_expression.txt, symbolic_regression_metrics.json,
#          symbolic_regression_report.md in each mldrive0/nnK and mldrive1/nnK dir.
#
# Usage: ./run_symbolic_regression.sh [data_directory] [-max_samples N] [-generations N] ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-$SCRIPT_DIR/../../../data}"

echo "Data directory: $DATA_DIR"
echo ""

python3 "$SCRIPT_DIR/symbolic_regression.py" "$DATA_DIR" "${@:2}"
