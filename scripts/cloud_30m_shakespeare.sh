#!/bin/bash
# 30M parameter char model on Shakespeare — cloud GPU run
# Too slow on laptop, ~5 minutes on A100
#
# Usage: bash scripts/cloud_30m_shakespeare.sh

set -e
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════"
echo "  Cloud: 30M Shakespeare (3000 steps)"
echo "═══════════════════════════════════════════"
echo ""

python3 -c "
import sys, os, time, torch
os.environ['PYTHONUNBUFFERED'] = '1'
sys.path.insert(0, '.')
from model.gpt import GPT

device = 'cuda'

text = open('data/input.txt').read()
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(len(data) * 0.9)
train_data, val_data = data[:n], data[n:]

model = GPT(
    vocab_size=len(chars), block_size=512,
    n_layer=8, n_head=8, n_embd=512,
).to(device)
print(f'Model: {model.n_params:,} params on {device}')

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
block_size, batch_size = 512, 64

t0 = time.time()
best_val = float('inf')
for step in range(1, 3001):
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
        idx = torch.tensor([[stoi['\n']]], device=device)
        sample = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
        text_out = ''.join([itos[i] for i in sample[0].tolist()])
        print(f'step {step} | train {loss.item():.4f} | val {vl:.4f} | {tok_s:,.0f} tok/s')
        print(f'  sample: {repr(text_out[:150])}')
        print()
        model.train()

dt = time.time() - t0
print(f'Done in {dt:.1f}s ({dt/60:.1f}m)')
print(f'Best val loss: {best_val:.4f}')
print()
print('--- Final sample (longer) ---')
idx = torch.tensor([[stoi['\n']]], device=device)
sample = model.generate(idx, max_new_tokens=500, temperature=0.8, top_k=40)
print(''.join([itos[i] for i in sample[0].tolist()]))
"
