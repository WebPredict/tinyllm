#!/bin/bash
# Quick test training on cloud GPU
# Runs a small model for 500 steps to verify everything works.
# Should complete in 2-3 minutes on an A100.
#
# Usage: bash scripts/cloud_test_train.sh

set -e
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════"
echo "  Cloud Test: Quick 500-step training run"
echo "═══════════════════════════════════════════"
echo ""

python3 -c "
import sys, os, time, torch

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, '.')
from model.gpt import GPT

# Auto-detect device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Device: {device}')

# Check if BPE data exists, fall back to Shakespeare
bpe_corpus = 'data/react-ts/corpus.txt'
bpe_tokenizer = 'data/react-ts/tokenizer/bpe_8000.json'

if os.path.exists(bpe_corpus) and os.path.exists(bpe_tokenizer):
    print('Using React/TS BPE data')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(bpe_tokenizer)
    text = open(bpe_corpus).read()
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    vocab_size = tokenizer.get_vocab_size()
    is_bpe = True
else:
    print('BPE data not found, using Shakespeare char-level')
    text = open('data/input.txt').read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    vocab_size = len(chars)
    is_bpe = False

n = int(len(data) * 0.95)
train_data = data[:n]
print(f'Vocab size: {vocab_size:,}')
print(f'Training tokens: {len(train_data):,}')

# Small model
block_size = 256
batch_size = 64
model = GPT(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=6,
    n_head=4,
    n_embd=256,
    dropout=0.1,
).to(device)

print(f'Model: {model.n_params:,} params on {device}')
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print()
print('Training 500 steps...')
t0 = time.time()
for step in range(1, 501):
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([train_data[i+1:i+1+block_size] for i in ix]).to(device)

    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        dt = time.time() - t0
        tok_s = (step * batch_size * block_size) / dt
        print(f'  step {step:>4} | loss {loss.item():.4f} | {tok_s:,.0f} tok/s', flush=True)

dt = time.time() - t0
tok_s = (500 * batch_size * block_size) / dt

print()
print(f'Done in {dt:.1f}s')
print(f'Throughput: {tok_s:,.0f} tokens/sec')
print()

# Compare to laptop
laptop_tok_s = 21000
speedup = tok_s / laptop_tok_s
print(f'vs laptop (~{laptop_tok_s:,} tok/s): {speedup:.1f}x faster')
print()

# Estimate training times on this hardware
print('Estimated training times on this GPU:')
for name, tokens in [('100M model, 2B tokens', 2e9), ('300M model, 5B tokens', 5e9), ('1B model, 15B tokens', 15e9), ('3B model, 60B tokens', 60e9)]:
    hours = tokens / tok_s / 3600
    cost = hours * 1.5
    print(f'  {name}: ~{hours:.1f} hours (~\${cost:.0f})')

# Generate a sample
print()
idx = torch.tensor([[0]], dtype=torch.long, device=device)
output = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=40)
if is_bpe:
    sample = tokenizer.decode(output[0].tolist())
else:
    itos = {i: ch for i, ch in enumerate(chars)}
    sample = ''.join([itos[i] for i in output[0].tolist()])
print('Sample output:')
print(sample[:300])
"

echo ""
echo "═══════════════════════════════════════════"
echo "  Test complete! Check throughput above."
echo "  If it's 5-10x faster than laptop, GPU is working."
echo "  REMEMBER: Shut down the pod when done!"
echo "═══════════════════════════════════════════"
