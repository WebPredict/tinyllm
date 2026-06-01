#!/bin/bash
# Train a 16K vocab BPE tokenizer on the Stack frontend corpus
# Usage: bash scripts/train_tokenizer_stack.sh

set -e
cd "$(dirname "$0")/.."

pip install tokenizers --quiet 2>/dev/null

python3 scripts/train_tokenizer_16k.py
