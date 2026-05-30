"""
Training loop for token-level (BPE) GPT.
Uses the trained BPE tokenizer instead of character-level encoding.
"""

import os
import sys
import json
import time
import sqlite3
import torch
from pathlib import Path
from tokenizers import Tokenizer

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.gpt import GPT


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

config = {
    # Model
    "n_layer": 6,
    "n_head": 4,
    "n_embd": 256,
    "block_size": 256,
    "dropout": 0.1,

    # Training
    "batch_size": 64,
    "learning_rate": 5e-4,
    "max_steps": 5000,
    "eval_interval": 250,
    "eval_steps": 20,
    "log_interval": 50,
    "checkpoint_interval": 1000,

    # Data
    "data_file": "data/react-ts/corpus.txt",
    "tokenizer_file": "data/react-ts/tokenizer/bpe_8000.json",
    "train_split": 0.95,

    # System
    "device": "auto",
    "run_id": "run_5m_react_bpe",
}


def get_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return requested


def load_data(data_file, tokenizer_file, train_split):
    """Load and tokenize data using BPE tokenizer."""
    root = Path(__file__).parent.parent

    # Load tokenizer
    tok_path = root / tokenizer_file
    print(f"Tokenizer: {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab_size = tokenizer.get_vocab_size()

    # Load and encode text
    data_path = root / data_file
    print(f"Data: {data_path}")
    text = data_path.read_text(encoding="utf-8")

    print(f"Encoding {len(text):,} characters...")
    encoded = tokenizer.encode(text)
    tokens = encoded.ids
    data = torch.tensor(tokens, dtype=torch.long)

    # Split
    n = int(len(data) * train_split)
    train_data = data[:n]
    val_data = data[n:]

    compression = len(text) / len(tokens)

    print(f"Vocab size: {vocab_size:,}")
    print(f"Total tokens: {len(tokens):,} (from {len(text):,} chars)")
    print(f"Compression: {compression:.1f} chars/token")
    print(f"Train: {len(train_data):,} tokens, Val: {len(val_data):,} tokens")

    return train_data, val_data, vocab_size, tokenizer, compression


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_steps, batch_size, block_size, device):
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        batch_losses = []
        for _ in range(eval_steps):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[name] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


def generate_sample(model, tokenizer, device, max_tokens=100, prompt="\n"):
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    if idx.shape[1] == 0:
        idx = torch.tensor([[0]], dtype=torch.long, device=device)

    output = model.generate(idx, max_new_tokens=max_tokens, temperature=0.8, top_k=40)
    return tokenizer.decode(output[0].tolist())


def init_db(db_path):
    db = sqlite3.connect(db_path)
    db.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            run_id TEXT, step INTEGER, timestamp REAL,
            train_loss REAL, val_loss REAL, learning_rate REAL,
            tokens_seen INTEGER, tokens_per_second REAL, sample_output TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, config TEXT, started REAL,
            finished REAL, status TEXT, final_train_loss REAL, final_val_loss REAL
        )
    """)
    db.commit()
    return db


def save_checkpoint(model, optimizer, step, cfg, run_id, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{run_id}_step{step}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": cfg,
        "run_id": run_id,
        "tokenizer_type": "bpe",
    }, path)
    print(f"  Checkpoint saved: {path}")


def train():
    device = get_device(config["device"])
    run_id = config["run_id"]
    print(f"Device: {device}")
    print(f"Run ID: {run_id}")
    print(f"Config: {json.dumps({k: v for k, v in config.items() if k != 'run_id'}, indent=2)}")
    print()

    # Data
    train_data, val_data, vocab_size, tokenizer, compression = load_data(
        config["data_file"], config["tokenizer_file"], config["train_split"]
    )
    config["vocab_size"] = vocab_size
    config["compression"] = compression
    print()

    # Model
    model = GPT(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
    ).to(device)

    print(f"Model: {model.n_params:,} parameters")
    print(f"  {config['n_layer']} layers, {config['n_head']} heads, {config['n_embd']} embd")
    print(f"  Block size: {config['block_size']} tokens (~{int(config['block_size'] * compression)} chars)")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # Logging
    db_path = Path(__file__).parent.parent / "logs" / "training.db"
    os.makedirs(db_path.parent, exist_ok=True)
    db = init_db(str(db_path))
    db.execute(
        "INSERT OR REPLACE INTO runs (run_id, config, started, status) VALUES (?, ?, ?, ?)",
        (run_id, json.dumps(config), time.time(), "running")
    )
    db.commit()

    checkpoint_dir = str(Path(__file__).parent.parent / "checkpoints")

    # Training loop
    print("Training...")
    print("=" * 60)

    t0 = time.time()
    tokens_seen = 0
    best_val_loss = float('inf')

    for step in range(1, config["max_steps"] + 1):
        x, y = get_batch(train_data, config["batch_size"], config["block_size"], device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tokens_seen += config["batch_size"] * config["block_size"]

        if step % config["log_interval"] == 0:
            dt = time.time() - t0
            tokens_per_sec = tokens_seen / dt
            print(f"  step {step:5d} | loss {loss.item():.4f} | {tokens_per_sec:,.0f} tok/s")

        if step % config["eval_interval"] == 0:
            losses = estimate_loss(
                model, train_data, val_data,
                config["eval_steps"], config["batch_size"], config["block_size"], device
            )
            sample = generate_sample(model, tokenizer, device)

            print()
            print(f"  ── Eval at step {step} ──")
            print(f"  Train loss: {losses['train']:.4f}")
            print(f"  Val loss:   {losses['val']:.4f}")
            print(f"  Sample: {repr(sample[:200])}")
            print()

            dt = time.time() - t0
            db.execute(
                """INSERT INTO training_logs
                   (run_id, step, timestamp, train_loss, val_loss,
                    learning_rate, tokens_seen, tokens_per_second, sample_output)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, step, time.time(), losses["train"], losses["val"],
                 config["learning_rate"], tokens_seen, tokens_seen / dt, sample[:500])
            )
            db.commit()

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        if step % config["checkpoint_interval"] == 0:
            save_checkpoint(model, optimizer, step, config, run_id, checkpoint_dir)

    # Final
    dt = time.time() - t0
    final_losses = estimate_loss(
        model, train_data, val_data,
        config["eval_steps"], config["batch_size"], config["block_size"], device
    )

    print()
    print("=" * 60)
    print(f"Training complete!")
    print(f"  Total time: {dt:.1f}s ({dt/60:.1f}m)")
    print(f"  Tokens seen: {tokens_seen:,}")
    print(f"  Final train loss: {final_losses['train']:.4f}")
    print(f"  Final val loss: {final_losses['val']:.4f}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Tokens/sec: {tokens_seen/dt:,.0f}")

    save_checkpoint(model, optimizer, step, config, run_id, checkpoint_dir)

    db.execute(
        """UPDATE runs SET finished=?, status=?, final_train_loss=?, final_val_loss=?
           WHERE run_id=?""",
        (time.time(), "complete", final_losses["train"], final_losses["val"], run_id)
    )
    db.commit()

    print()
    print("─── Final generated sample ───")
    print(generate_sample(model, tokenizer, device, max_tokens=200))


if __name__ == "__main__":
    train()
