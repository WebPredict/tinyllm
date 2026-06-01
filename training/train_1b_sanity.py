"""
1B sanity check — short run with frequent checkpoints and validation.
Verifies the full pipeline works before committing to a long run.
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.gpt import GPT


config = {
    # Model — 1B parameters
    "n_layer": 24,
    "n_head": 16,
    "n_embd": 2048,
    "block_size": 2048,
    "dropout": 0.1,

    # Training — conservative settings
    "batch_size": 8,
    "grad_accum_steps": 4,
    "learning_rate": 2e-4,       # lower than before (was 3e-4)
    "min_lr": 2e-5,
    "max_steps": 1000,
    "warmup_steps": 100,
    "eval_interval": 250,
    "log_interval": 50,
    "checkpoint_interval": 250,   # checkpoint every 250 steps

    # Data
    "corpus_file": "data/stack-frontend/corpus.txt",
    "tokenizer_file": "data/stack-frontend/tokenizer/bpe_16000.json",
    "train_split": 0.95,

    # System
    "run_id": "cloud_1b_sanity",
    "compile": False,             # disable torch.compile to avoid _orig_mod prefix
}


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        is_main = rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_main = True
    return rank, local_rank, world_size, device, is_main


def log(msg, is_main=True):
    if is_main:
        print(msg, flush=True)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def load_data(corpus_file, tokenizer_file, train_split, is_main):
    from tokenizers import Tokenizer
    log(f"Loading tokenizer: {tokenizer_file}", is_main)
    tokenizer = Tokenizer.from_file(tokenizer_file)
    vocab_size = tokenizer.get_vocab_size()

    log(f"Loading corpus: {corpus_file}", is_main)
    text = open(corpus_file).read()
    log(f"Encoding {len(text):,} characters...", is_main)
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    compression = len(text) / len(data)

    n = int(len(data) * train_split)
    train_data = data[:n]
    val_data = data[n:]

    log(f"Vocab: {vocab_size:,}, Tokens: {len(data):,}, Compression: {compression:.1f}x", is_main)
    log(f"Train: {len(train_data):,}, Val: {len(val_data):,}", is_main)
    return train_data, val_data, vocab_size, tokenizer, compression


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+1+block_size] for i in ix]).to(device)
    return x, y


def save_checkpoint(model, step, config, best_val_loss, is_main, world_size):
    """Save checkpoint and verify it's loadable."""
    if not is_main:
        return

    os.makedirs("checkpoints", exist_ok=True)

    if world_size > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    # Strip any _orig_mod prefix (from torch.compile)
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v

    path = f"checkpoints/{config['run_id']}_step{step}.pt"
    torch.save({
        "model_state_dict": cleaned,
        "config": {k: v for k, v in config.items() if k != "compile"},
        "step": step,
        "run_id": config["run_id"],
        "tokenizer_type": "bpe",
        "best_val_loss": best_val_loss,
    }, path)
    size_mb = os.path.getsize(path) / 1e6
    log(f"  Checkpoint saved: {path} ({size_mb:.0f}MB)")

    # Verify checkpoint is loadable
    try:
        test_cp = torch.load(path, map_location="cpu", weights_only=False)
        test_model = GPT(
            vocab_size=config["vocab_size"],
            block_size=config["block_size"],
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_embd=config["n_embd"],
        )
        test_model.load_state_dict(test_cp["model_state_dict"])

        # Check for NaN
        nan_count = sum(torch.isnan(v).sum().item() for v in test_cp["model_state_dict"].values() if torch.is_floating_point(v))
        if nan_count > 0:
            log(f"  WARNING: {nan_count} NaN values in checkpoint!")
        else:
            log(f"  Checkpoint verified: loadable, no NaN")
        del test_model, test_cp
    except Exception as e:
        log(f"  WARNING: Checkpoint verification failed: {e}")


def train():
    rank, local_rank, world_size, device, is_main = setup_distributed()

    log("=" * 60, is_main)
    log(f"  1B Model Sanity Check (1000 steps)", is_main)
    log(f"  GPUs: {world_size}, Device: {device}", is_main)
    log(f"  Learning rate: {config['learning_rate']} (conservative)", is_main)
    log(f"  Checkpoints every {config['checkpoint_interval']} steps", is_main)
    log(f"  torch.compile: {config['compile']}", is_main)
    log("=" * 60, is_main)
    log("", is_main)

    # Data
    train_data, val_data, vocab_size, tokenizer, compression = load_data(
        config["corpus_file"], config["tokenizer_file"],
        config["train_split"], is_main,
    )
    config["vocab_size"] = vocab_size

    # Model
    log("Building model...", is_main)
    model = GPT(
        vocab_size=vocab_size,
        block_size=config["block_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
    ).to(device)

    log(f"Model: {model.n_params:,} params", is_main)

    # FSDP for multi-GPU
    if world_size > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(model, mixed_precision=mixed_precision)
        log(f"Wrapped with FSDP ({world_size} GPUs, BF16)", is_main)
    else:
        log("Single GPU mode", is_main)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=0.1, betas=(0.9, 0.95),
    )

    # Training
    log("\nTraining...", is_main)
    log("=" * 60, is_main)

    t0 = time.time()
    best_val_loss = float("inf")
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    grad_accum = config["grad_accum_steps"]

    for step in range(1, config["max_steps"] + 1):
        lr = get_lr(step, config["warmup_steps"], config["max_steps"],
                    config["learning_rate"], config["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0

        for micro_step in range(grad_accum):
            x, y = get_batch(train_data, batch_size, block_size, device)
            if world_size > 1:
                _, loss = model(x, y)
            else:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        # Check for NaN loss BEFORE optimizer step
        if math.isnan(accum_loss):
            log(f"\n  FATAL: NaN loss at step {step}! Stopping.", is_main)
            log(f"  Last good checkpoint should be usable.", is_main)
            break

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # tighter clipping
        optimizer.step()

        tokens_seen = step * batch_size * block_size * grad_accum * world_size

        if step % config["log_interval"] == 0 and is_main:
            dt = time.time() - t0
            tok_s = tokens_seen / dt
            log(f"  step {step:>5} | loss {accum_loss:.4f} | lr {lr:.6f} | grad_norm {grad_norm:.2f} | {tok_s:,.0f} tok/s")

        # Eval
        if step % config["eval_interval"] == 0 and is_main:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(20):
                    x, y = get_batch(val_data, batch_size, block_size, device)
                    if world_size > 1:
                        _, loss = model(x, y)
                    else:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            _, loss = model(x, y)
                    val_loss += loss.item()
            val_loss /= 20
            best_val_loss = min(best_val_loss, val_loss)

            # Generate sample
            raw_model = model.module if world_size > 1 else model
            idx = torch.tensor([[0]], device=device)
            with torch.no_grad():
                sample_ids = raw_model.generate(idx, max_new_tokens=150, temperature=0.8, top_k=40)
            sample_text = tokenizer.decode(sample_ids[0].tolist())

            log(f"\n  ── Eval at step {step} ──")
            log(f"  Train loss: {accum_loss:.4f}")
            log(f"  Val loss:   {val_loss:.4f} (best: {best_val_loss:.4f})")
            log(f"  Sample: {repr(sample_text[:200])}")
            log("")
            model.train()

        # Checkpoint
        if step % config["checkpoint_interval"] == 0:
            save_checkpoint(model, step, config, best_val_loss, is_main, world_size)

    # Final
    dt = time.time() - t0
    if is_main:
        log("")
        log("=" * 60)
        log(f"Sanity check complete!")
        log(f"  Time: {dt/60:.1f} minutes")
        log(f"  Best val loss: {best_val_loss:.4f}")
        log(f"  Checkpoints saved at steps: 250, 500, 750, 1000")
        log("")
        log("  Verify on your laptop:")
        log("    scp the checkpoint, then:")
        log("    python -c \"import torch; cp = torch.load('checkpoint.pt', map_location='cpu'); print('OK')\"")
        log("=" * 60)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
