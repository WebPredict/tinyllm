#!/bin/bash
# Run a 5M parameter char model on Shakespeare
# Purpose: test scaling from 834K → 5M on laptop, measure throughput
#
# Expected: ~30-60 minutes on Mac, lower loss than the 834K model
#
# Usage: ./scripts/run_5m_train.sh

set -e
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════"
echo "  TinyLLM — 5M Parameter Run"
echo "  Model: ~5M params, char-level"
echo "  Data: Shakespeare (1MB)"
echo "  Purpose: Laptop scaling test"
echo "═══════════════════════════════════════════"
echo ""

source venv/bin/activate
python training/train_5m.py
