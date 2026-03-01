#!/bin/bash
# Run the activation insights agent on activation maximization data.
#
# Prerequisites:
#   1. Activation maximization has been run (run_activation_maximization_all.sh)
#   2. For LLM mode: Install Ollama (https://ollama.com) and run: ollama run llama3.2
#   3. For template mode: pip install -r requirements.txt (no Ollama needed)
#
# Usage:
#   ./run_activation_insights_agent.sh [path]
#
# Examples:
#   # Process single activation_maximization directory
#   ./run_activation_insights_agent.sh /path/to/activation_maximization
#
#   # Search entire data directory for all activation_maximization folders
#   ./run_activation_insights_agent.sh --search ../../data
#
#   # Use template output (no LLM, no Ollama required)
#   python3 activation_insights_agent.py /path/to/activation_maximization --no-llm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default data root: integration/client-level/data
DATA_ROOT="${1:-$SCRIPT_DIR/../../../data}"

echo "=========================================="
echo "Activation Insights Agent"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo ""

# Search for activation_maximization dirs and process each
python3 "$SCRIPT_DIR/activation_insights_agent.py" "$DATA_ROOT" --search
