#!/bin/bash
# Analyze and optionally filter the React/TS training data
# Usage:
#   ./scripts/run_analyze_data.sh           ← analyze only
#   ./scripts/run_analyze_data.sh --filter  ← analyze + filter

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "Step 1: Analyzing data quality..."
python scripts/analyze_data.py

if [ "$1" = "--filter" ]; then
    echo ""
    echo "Step 2: Filtering corpus..."
    python scripts/filter_data.py
fi
