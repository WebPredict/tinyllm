#!/bin/bash
# 30M parameter BPE model on React/TS data — cloud GPU run
# Requires: data/react-ts/corpus.txt and data/react-ts/tokenizer/bpe_8000.json
# Download from GitHub release if not present:
#   mkdir -p data/react-ts/tokenizer
#   wget https://github.com/WebPredict/tinyllm/releases/download/v0.1-data/corpus.txt -P data/react-ts/
#   wget https://github.com/WebPredict/tinyllm/releases/download/v0.1-data/bpe_8000.json -P data/react-ts/tokenizer/
#
# Usage: bash scripts/cloud_30m_react_bpe.sh

set -e
cd "$(dirname "$0")/.."

# Check data exists
if [ ! -f "data/react-ts/corpus.txt" ]; then
    echo "React/TS corpus not found. Downloading from GitHub release..."
    mkdir -p data/react-ts/tokenizer
    wget -q https://github.com/WebPredict/tinyllm/releases/download/v0.1-data/corpus.txt -P data/react-ts/
    wget -q https://github.com/WebPredict/tinyllm/releases/download/v0.1-data/bpe_8000.json -P data/react-ts/tokenizer/
fi

# Check tokenizers is installed
pip install tokenizers --quiet 2>/dev/null

echo "═══════════════════════════════════════════"
echo "  Cloud: 30M React/TS BPE (5000 steps)"
echo "═══════════════════════════════════════════"
echo ""

python3 -c "
import sys, os, time, torch
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, '.')
from model.gpt import GPT
from tokenizers import Tokenizer

device = 'cuda'

# Load tokenizer and data
tokenizer = Tokenizer.from_file('data/react-ts/tokenizer/bpe_8000.json')
text = open('data/react-ts/corpus.txt').read()
print(f'Encoding {len(text):,} characters...')
encoded = tokenizer.encode(text)
data = torch.tensor(encoded.ids, dtype=torch.long)
vocab_size = tokenizer.get_vocab_size()
n = int(len(data) * 0.95)
train_data, val_data = data[:n], data[n:]
compression = len(text) / len(data)
print(f'Vocab: {vocab_size:,}, Tokens: {len(data):,}, Compression: {compression:.1f}x')
print(f'Train: {len(train_data):,}, Val: {len(val_data):,}')

# 30M model
block_size = 512
batch_size = 64
model = GPT(
    vocab_size=vocab_size, block_size=block_size,
    n_layer=8, n_head=8, n_embd=512, dropout=0.1,
).to(device)
print(f'Model: {model.n_params:,} params on {device}')
print(f'Effective context: ~{int(block_size * compression)} characters')
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

t0 = time.time()
best_val = float('inf')
for step in range(1, 5001):
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([train_data[i+1:i+1+block_size] for i in ix]).to(device)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        model.eval()
        vl = 0
        with torch.no_grad():
            for _ in range(20):
                ix = torch.randint(len(val_data) - block_size, (batch_size,))
                x = torch.stack([val_data[i:i+block_size] for i in ix]).to(device)
                y = torch.stack([val_data[i+1:i+1+block_size] for i in ix]).to(device)
                _, l = model(x, y)
                vl += l.item()
        vl /= 20
        best_val = min(best_val, vl)
        dt = time.time() - t0
        tok_s = (step * batch_size * block_size) / dt
        idx = torch.tensor([[0]], device=device)
        sample = model.generate(idx, max_new_tokens=150, temperature=0.8, top_k=40)
        text_out = tokenizer.decode(sample[0].tolist())
        print(f'step {step} | train {loss.item():.4f} | val {vl:.4f} | {tok_s:,.0f} tok/s')
        print(f'  sample: {repr(text_out[:200])}')
        print()
        model.train()

dt = time.time() - t0
print(f'Done in {dt:.1f}s ({dt/60:.1f}m)')
print(f'Best val loss: {best_val:.4f}')
print(f'Perplexity: {2.718**best_val:.2f}')
print()
print('--- Final sample ---')
idx = torch.tensor([[0]], device=device)
sample = model.generate(idx, max_new_tokens=300, temperature=0.8, top_k=40)
print(tokenizer.decode(sample[0].tolist()))

# Save checkpoint
import os
os.makedirs('checkpoints', exist_ok=True)
checkpoint_path = 'checkpoints/cloud_30m_react_bpe.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': 8,
        'n_head': 8,
        'n_embd': 512,
        'dropout': 0.1,
        'data_file': 'data/react-ts/corpus.txt',
        'tokenizer_file': 'data/react-ts/tokenizer/bpe_8000.json',
        'train_split': 0.95,
    },
    'step': 5000,
    'run_id': 'cloud_30m_react_bpe',
    'tokenizer_type': 'bpe',
    'best_val_loss': best_val,
}, checkpoint_path)
print(f'Checkpoint saved: {checkpoint_path}')
print(f'Size: {os.path.getsize(checkpoint_path) / 1e6:.1f} MB')
print()
print('To download to your laptop:')
print(f'  scp -P <port> root@<ip>:tinyllm/{checkpoint_path} ./checkpoints/')
"
