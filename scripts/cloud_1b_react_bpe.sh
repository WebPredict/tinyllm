#!/bin/bash
# 1B parameter BPE model on frontend data — 4x H100 cloud run
#
# PREREQUISITES:
#   1. Large corpus downloaded (2-5GB, ~600M-1.5B tokens)
#      Either via get_stack_data.py or uploaded to GitHub release
#   2. RunPod pod with 4x H100 GPUs
#   3. pip install tokenizers torch
#
# Expected: ~6 hours on 4x H100, ~$78
#
# Usage: bash scripts/cloud_run.sh scripts/cloud_1b_react_bpe.sh

set -e
cd "$(dirname "$0")/.."

# Check/download data
DATA_DIR="data/stack-frontend"
CORPUS="$DATA_DIR/corpus.txt"
TOKENIZER="$DATA_DIR/tokenizer/bpe_16000.json"

if [ ! -f "$CORPUS" ]; then
    echo "Large corpus not found at $CORPUS"
    echo ""
    echo "Options:"
    echo "  1. Download from The Stack (needs HuggingFace account):"
    echo "     pip install datasets huggingface_hub"
    echo "     huggingface-cli login"
    echo "     python scripts/get_stack_data.py --target-gb 3"
    echo ""
    echo "  2. Download from GitHub release (if uploaded):"
    echo "     mkdir -p $DATA_DIR"
    echo "     wget https://github.com/WebPredict/tinyllm/releases/download/v0.3-data/corpus_stack.txt -O $CORPUS"
    exit 1
fi

# Train tokenizer if needed (larger vocab for 1B model)
if [ ! -f "$TOKENIZER" ]; then
    echo "Training 16K vocab tokenizer..."
    pip install tokenizers --quiet 2>/dev/null
    mkdir -p "$DATA_DIR/tokenizer"
    python3 -c "
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
t.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=16000, special_tokens=['<|pad|>','<|endoftext|>','<|startoftext|>'], min_frequency=2, show_progress=True)
t.train(['$CORPUS'], trainer)
t.save('$TOKENIZER')
print(f'Tokenizer saved: {t.get_vocab_size()} vocab')
"
fi

echo "═══════════════════════════════════════════"
echo "  Cloud: 1B React/TS BPE (4x H100)"
echo "═══════════════════════════════════════════"
echo ""

# Check GPU count
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'GPUs available: {n}')
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)')
if n < 2:
    print('WARNING: Only 1 GPU detected. Training will be slow.')
    print('For 1B model, 4x H100 is recommended.')
"

echo ""

# Launch distributed training
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training on $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS training/train_distributed.py
else
    echo "Single GPU mode..."
    python3 training/train_distributed.py
fi
