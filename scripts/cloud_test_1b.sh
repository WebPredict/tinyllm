#!/bin/bash
# Quick throughput test for 1B model on multi-GPU
# Runs 200 steps to measure actual tok/s, then estimates full training time.
# Should take 5-10 minutes. Cost: ~$1
#
# Usage: bash scripts/cloud_test_1b.sh

set -e
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════"
echo "  1B Model Throughput Test (200 steps)"
echo "═══════════════════════════════════════════"
echo ""

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "GPUs detected: $NUM_GPUS"
echo ""

# Need some data — use whatever is available
if [ -f "data/stack-frontend/corpus.txt" ]; then
    CORPUS="data/stack-frontend/corpus.txt"
    TOKENIZER="data/stack-frontend/tokenizer/bpe_16000.json"
elif [ -f "data/react-ts-expanded/corpus.txt" ]; then
    CORPUS="data/react-ts-expanded/corpus.txt"
    TOKENIZER="data/react-ts-expanded/tokenizer/bpe_8000.json"
elif [ -f "data/react-ts/corpus.txt" ]; then
    CORPUS="data/react-ts/corpus.txt"
    TOKENIZER="data/react-ts/tokenizer/bpe_8000.json"
else
    echo "No corpus found. Downloading expanded data..."
    mkdir -p data/react-ts-expanded/tokenizer
    wget -q https://github.com/WebPredict/tinyllm/releases/download/v0.2-data/corpus.txt -O data/react-ts-expanded/corpus.txt
    wget -q https://github.com/WebPredict/tinyllm/releases/download/v0.2-data/bpe_8000.json -O data/react-ts-expanded/tokenizer/bpe_8000.json
    CORPUS="data/react-ts-expanded/corpus.txt"
    TOKENIZER="data/react-ts-expanded/tokenizer/bpe_8000.json"
fi

echo "Corpus: $CORPUS"
echo "Tokenizer: $TOKENIZER"
echo ""

# Write test script
python3 -c "
import os, sys, time, math, torch
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, '.')
from model.gpt import GPT
from tokenizers import Tokenizer

device = 'cuda'
num_gpus = torch.cuda.device_count()

# Load data
tokenizer = Tokenizer.from_file('$TOKENIZER')
text = open('$CORPUS').read()[:5_000_000]  # only need a small chunk for testing
encoded = tokenizer.encode(text)
data = torch.tensor(encoded.ids, dtype=torch.long)
vocab_size = tokenizer.get_vocab_size()
print(f'Data: {len(data):,} tokens')

# 1B model config
n_layer, n_head, n_embd = 24, 16, 2048
block_size = 2048
batch_size = 8
grad_accum = 4

# Test different block sizes
for bs in [1024, 2048]:
    print(f'\n--- Block size: {bs}, Batch: {batch_size}, Grad accum: {grad_accum} ---')

    model = GPT(
        vocab_size=vocab_size, block_size=bs,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=0.1,
    ).to(device)

    # BF16
    model = model.to(torch.bfloat16)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params:,} params on {device} (bf16)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Warmup
    print('Warming up...')
    for _ in range(5):
        ix = torch.randint(len(data) - bs, (batch_size,))
        x = torch.stack([data[i:i+bs] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+1+bs] for i in ix]).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Timed run
    print('Running 200 steps...')
    torch.cuda.synchronize()
    t0 = time.time()
    total_tokens = 0

    for step in range(1, 201):
        optimizer.zero_grad()
        for _ in range(grad_accum):
            ix = torch.randint(len(data) - bs, (batch_size,))
            x = torch.stack([data[i:i+bs] for i in ix]).to(device)
            y = torch.stack([data[i+1:i+1+bs] for i in ix]).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            total_tokens += batch_size * bs

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            torch.cuda.synchronize()
            dt = time.time() - t0
            tok_s = total_tokens / dt
            print(f'  step {step:>4} | loss {loss.item():.4f} | {tok_s:,.0f} tok/s')

    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_s = total_tokens / dt

    print(f'\nResults (block_size={bs}):')
    print(f'  Throughput: {tok_s:,.0f} tok/s (single GPU)')
    print(f'  With {num_gpus}x GPU (estimated): {tok_s * num_gpus * 0.85:,.0f} tok/s')
    print()

    # Estimate training times
    effective_tok_s = tok_s * num_gpus * 0.85  # 85% scaling efficiency
    print(f'  Estimated training times ({num_gpus}x GPU):')
    for name, tokens in [('2B tokens', 2e9), ('5B tokens', 5e9), ('10B tokens', 10e9), ('15B tokens', 15e9), ('20B tokens', 20e9)]:
        hours = tokens / effective_tok_s / 3600
        cost = hours * 13  # 4x H100 rate
        print(f'    {name}: {hours:.1f} hours (\${cost:.0f})')

    del model, optimizer
    torch.cuda.empty_cache()

print('\n' + '=' * 60)
print('  Use these numbers to decide training token count')
print('  and adjust config in training/train_distributed.py')
print('=' * 60)
"
