#!/bin/bash
# 1B model sanity check — 1000 steps with frequent checkpoints
# Verifies: training works, checkpoints are usable, no NaN
# Expected: ~30 min on 4x H100, ~$7
#
# Usage: bash scripts/cloud_run.sh scripts/cloud_1b_sanity_check.sh

set -e
cd "$(dirname "$0")/.."

CORPUS="data/stack-frontend/corpus.txt"
TOKENIZER="data/stack-frontend/tokenizer/bpe_16000.json"

if [ ! -f "$CORPUS" ] || [ ! -f "$TOKENIZER" ]; then
    echo "Data not found. Download first:"
    echo "  python scripts/get_stack_data.py --target-gb 1.5"
    echo "  bash scripts/train_tokenizer_stack.sh"
    exit 1
fi

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")

echo "═══════════════════════════════════════════"
echo "  1B Sanity Check (1000 steps)"
echo "  GPUs: $NUM_GPUS"
echo "  Checkpoints every 250 steps"
echo "═══════════════════════════════════════════"
echo ""

if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$NUM_GPUS training/train_1b_sanity.py
else
    python3 training/train_1b_sanity.py
fi
