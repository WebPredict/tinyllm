#!/bin/bash
# Run a 5M parameter char model on React/TypeScript data
# Purpose: First domain-specific training run
#
# Key differences from Shakespeare runs:
#   - 52MB of data (vs 1MB) — should prevent overfitting
#   - Dropout 0.1 added as extra safety
#   - 95/5 train/val split (more data to train on)
#
# Usage: ./scripts/run_5m_react.sh

set -e
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════"
echo "  TinyLLM — 5M React/TypeScript Run"
echo "  Model: ~5M params, char-level"
echo "  Data: React/TS corpus (52MB)"
echo "  Purpose: First domain-specific training"
echo "═══════════════════════════════════════════"
echo ""

source venv/bin/activate
python training/train_5m_react.py
