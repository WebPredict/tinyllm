#!/bin/bash
# 100M parameter BPE model on expanded React/TS data — cloud GPU run
# Expected: ~1.5-2 hours on A100, ~$3
#
# Requires data uploaded to GitHub release first.
#
# Usage: bash scripts/cloud_100m_react_bpe.sh

set -e
cd "$(dirname "$0")/.."

# Check/download data
if [ ! -f "data/react-ts-expanded/corpus.txt" ]; then
    echo "Downloading expanded corpus from GitHub release..."
    mkdir -p data/react-ts-expanded/tokenizer
    wget -q https://github.com/WebPredict/tinyllm/releases/download/v0.2-data/corpus.txt -O data/react-ts-expanded/corpus.txt
    wget -q https://github.com/WebPredict/tinyllm/releases/download/v0.2-data/bpe_8000.json -O data/react-ts-expanded/tokenizer/bpe_8000.json
fi

if [ ! -f "data/react-ts-expanded/tokenizer/bpe_8000.json" ]; then
    echo "Training tokenizer on expanded data..."
    pip install tokenizers --quiet 2>/dev/null
    python3 -c "
from scripts.train_tokenizer import train_tokenizer, analyze_tokenizer
from pathlib import Path
t, p = train_tokenizer(Path('data/react-ts-expanded/corpus.txt'), vocab_size=8000)
analyze_tokenizer(t, Path('data/react-ts-expanded/corpus.txt'), 8000)
" 2>/dev/null || python3 scripts/train_tokenizer.py
fi

echo "═══════════════════════════════════════════"
echo "  Cloud: 100M React/TS BPE (10000 steps)"
echo "═══════════════════════════════════════════"
echo ""

python3 -c "
import sys, os, time, torch, json
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, '.')
from model.gpt import GPT
from tokenizers import Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer and data
tok_path = 'data/react-ts-expanded/tokenizer/bpe_8000.json'

tokenizer = Tokenizer.from_file(tok_path)
text = open('data/react-ts-expanded/corpus.txt').read()
print(f'Encoding {len(text):,} characters...')
encoded = tokenizer.encode(text)
data = torch.tensor(encoded.ids, dtype=torch.long)
vocab_size = tokenizer.get_vocab_size()
n = int(len(data) * 0.95)
train_data, val_data = data[:n], data[n:]
compression = len(text) / len(data)
print(f'Vocab: {vocab_size:,}, Tokens: {len(data):,}, Compression: {compression:.1f}x')
print(f'Train: {len(train_data):,}, Val: {len(val_data):,}')

# 100M model
block_size = 1024
batch_size = 32
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

model = GPT(
    vocab_size=vocab_size, block_size=block_size,
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout,
).to(device)
print(f'Model: {model.n_params:,} params on {device}')
print(f'Effective context: ~{int(block_size * compression)} characters')
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
max_steps = 10000
eval_interval = 500
checkpoint_interval = 2000

t0 = time.time()
best_val = float('inf')
tokens_per_step = batch_size * block_size

# Learning rate warmup + cosine decay
def get_lr(step):
    warmup_steps = 500
    if step < warmup_steps:
        return 3e-4 * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return 3e-4 * 0.5 * (1 + __import__('math').cos(__import__('math').pi * decay_ratio))

print(f'Training {max_steps} steps...')
print('=' * 60)
for step in range(1, max_steps + 1):
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([train_data[i+1:i+1+block_size] for i in ix]).to(device)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 100 == 0:
        dt = time.time() - t0
        tok_s = (step * tokens_per_step) / dt
        print(f'  step {step:>5} | loss {loss.item():.4f} | lr {lr:.6f} | {tok_s:,.0f} tok/s', flush=True)

    if step % eval_interval == 0:
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
        idx = torch.tensor([[0]], device=device)
        sample = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
        text_out = tokenizer.decode(sample[0].tolist())
        print(f'\n  ── Eval at step {step} ──')
        print(f'  Train loss: {loss.item():.4f}')
        print(f'  Val loss:   {vl:.4f} (best: {best_val:.4f})')
        print(f'  Sample: {repr(text_out[:200])}')
        print(flush=True)
        model.train()

    if step % checkpoint_interval == 0:
        os.makedirs('checkpoints', exist_ok=True)
        path = f'checkpoints/cloud_100m_react_bpe_step{step}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'vocab_size': vocab_size, 'block_size': block_size,
                'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd,
                'dropout': dropout,
                'data_file': 'data/react-ts-expanded/corpus.txt',
                'tokenizer_file': tok_path,
                'train_split': 0.95,
            },
            'step': step,
            'run_id': 'cloud_100m_react_bpe',
            'tokenizer_type': 'bpe',
            'best_val_loss': best_val,
        }, path)
        print(f'  Checkpoint saved: {path}', flush=True)

dt = time.time() - t0
print()
print('=' * 60)
print(f'Training complete!')
print(f'  Total time: {dt:.1f}s ({dt/60:.1f}m)')
print(f'  Best val loss: {best_val:.4f}')
print(f'  Perplexity: {2.718**best_val:.2f}')
print(f'  Tokens/sec: {(max_steps * tokens_per_step)/dt:,.0f}')

# Save final checkpoint
os.makedirs('checkpoints', exist_ok=True)
path = 'checkpoints/cloud_100m_react_bpe.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': vocab_size, 'block_size': block_size,
        'n_layer': n_layer, 'n_head': n_head, 'n_embd': n_embd,
        'dropout': dropout,
        'data_file': 'data/react-ts-expanded/corpus.txt',
        'tokenizer_file': tok_path,
        'train_split': 0.95,
    },
    'step': max_steps,
    'run_id': 'cloud_100m_react_bpe',
    'tokenizer_type': 'bpe',
    'best_val_loss': best_val,
}, path)
size_mb = os.path.getsize(path) / 1e6
print(f'  Checkpoint: {path} ({size_mb:.0f}MB)')
print()
print('--- Final sample ---')
idx = torch.tensor([[0]], device=device)
sample = model.generate(idx, max_new_tokens=500, temperature=0.8, top_k=40)
print(tokenizer.decode(sample[0].tolist()))
print()
print('To download: scp -P <port> root@<ip>:tinyllm/checkpoints/cloud_100m_react_bpe.pt ./checkpoints/')
"
