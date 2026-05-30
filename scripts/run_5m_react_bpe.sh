#!/bin/bash
# Run 5M model on React/TS data with BPE tokenizer
# Key comparison: same model size, same data, BPE vs character tokenization
#
# Usage: ./scripts/run_5m_react_bpe.sh

set -e
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════"
echo "  TinyLLM — 5M React/TS with BPE Tokenizer"
echo "  Comparing: BPE (8K vocab) vs char (1587 vocab)"
echo "  Same model size, same data, different encoding"
echo "═══════════════════════════════════════════"
echo ""

source venv/bin/activate
python training/train_bpe.py
