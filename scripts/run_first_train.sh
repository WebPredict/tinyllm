#!/bin/bash
# Run the first training experiment: 5M char model on Shakespeare
# This is the "does it work at all?" sanity check.
#
# Expected: ~10-20 minutes on Mac, loss drops from ~4.5 to ~1.5,
# generated text goes from random characters to Shakespeare-ish.
#
# Usage: ./scripts/run_first_train.sh

set -e
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════"
echo "  TinyLLM — First Training Run"
echo "  Model: ~5M params, char-level"
echo "  Data: Shakespeare (1MB)"
echo "  Purpose: Sanity check pipeline"
echo "═══════════════════════════════════════════"
echo ""

# Activate venv
source venv/bin/activate

# Download data if needed
echo "Step 1: Checking data..."
python scripts/get_shakespeare.py
echo ""

# Run training
echo "Step 2: Training..."
echo "(This should take 10-20 minutes on a Mac)"
echo ""
python training/train.py

echo ""
echo "═══════════════════════════════════════════"
echo "  Done! Check logs/training.db for metrics"
echo "═══════════════════════════════════════════"
