"""
Distributed training for 1B+ models using PyTorch FSDP.
Supports multi-GPU training on 2-8 GPUs.

Single GPU: python training/train_distributed.py
Multi GPU:  torchrun --nproc_per_node=4 training/train_distributed.py

Config is set for 1B model by default. Adjust config dict below for other sizes.
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.gpt import GPT

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

config = {
    # Model — 1B parameters
    "n_layer": 24,
    "n_head": 16,
    "n_embd": 2048,
    "block_size": 2048,
    "dropout": 0.1,

    # Training
    "batch_size": 8,           # per GPU
    "grad_accum_steps": 4,     # effective batch = 8 * 4 * num_gpus
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "max_steps": 50000,
    "warmup_steps": 1000,
    "eval_interval": 1000,
    "eval_steps": 20,
    "log_interval": 100,
    "checkpoint_interval": 5000,

    # Data
    "data_dir": "data/stack-frontend",
    "corpus_file": "data/stack-frontend/corpus.txt",
    "tokenizer_file": "data/stack-frontend/tokenizer/bpe_16000.json",
    "train_split": 0.95,

    # System
    "run_id": "cloud_1b_react_bpe",
    "compile": True,  # torch.compile for speed (PyTorch 2.0+)
}


def setup_distributed():
    """Initialize distributed training if available."""
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
    """Only print from the main process."""
    if is_main:
        print(msg, flush=True)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Learning rate schedule: linear warmup + cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def load_data(corpus_file, tokenizer_file, train_split, is_main):
    """Load and tokenize the corpus."""
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


def train():
    rank, local_rank, world_size, device, is_main = setup_distributed()

    log("=" * 60, is_main)
    log(f"  1B Model Distributed Training", is_main)
    log(f"  GPUs: {world_size}, Device: {device}", is_main)
    log(f"  Effective batch: {config['batch_size'] * config['grad_accum_steps'] * world_size}", is_main)
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
    log(f"  {config['n_layer']} layers, {config['n_head']} heads, {config['n_embd']} embd", is_main)
    log(f"  Block size: {config['block_size']} tokens (~{int(config['block_size'] * compression)} chars)", is_main)

    # Compile for speed
    if config["compile"] and hasattr(torch, "compile"):
        log("Compiling model with torch.compile...", is_main)
        model = torch.compile(model)

    # Wrap with FSDP for multi-GPU
    if world_size > 1:
        log(f"Wrapping with FSDP ({world_size} GPUs)...", is_main)
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        model = FSDP(model, mixed_precision=mixed_precision)
    else:
        # Single GPU — use BF16 autocast
        log("Single GPU mode with BF16 autocast", is_main)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # Training loop
    log("", is_main)
    log("Training...", is_main)
    log("=" * 60, is_main)

    t0 = time.time()
    best_val_loss = float("inf")
    tokens_seen = 0
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    grad_accum = config["grad_accum_steps"]

    for step in range(1, config["max_steps"] + 1):
        # Update learning rate
        lr = get_lr(step, config["warmup_steps"], config["max_steps"],
                    config["learning_rate"], config["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tokens_seen += batch_size * block_size * grad_accum * world_size

        # Logging
        if step % config["log_interval"] == 0 and is_main:
            dt = time.time() - t0
            tok_s = tokens_seen / dt
            log(f"  step {step:>6} | loss {accum_loss:.4f} | lr {lr:.6f} | {tok_s:,.0f} tok/s")

        # Evaluation
        if step % config["eval_interval"] == 0 and is_main:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(config["eval_steps"]):
                    x, y = get_batch(val_data, batch_size, block_size, device)
                    if world_size > 1:
                        _, loss = model(x, y)
                    else:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            _, loss = model(x, y)
                    val_loss += loss.item()
            val_loss /= config["eval_steps"]
            best_val_loss = min(best_val_loss, val_loss)

            # Generate sample
            raw_model = model.module if world_size > 1 else model
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod  # unwrap torch.compile
            idx = torch.tensor([[0]], device=device)
            with torch.no_grad():
                sample_ids = raw_model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
            sample_text = tokenizer.decode(sample_ids[0].tolist())

            log(f"\n  ── Eval at step {step} ──")
            log(f"  Train loss: {accum_loss:.4f}")
            log(f"  Val loss:   {val_loss:.4f} (best: {best_val_loss:.4f})")
            log(f"  Sample: {repr(sample_text[:200])}")
            log("")
            model.train()

        # Checkpoint
        if step % config["checkpoint_interval"] == 0 and is_main:
            os.makedirs("checkpoints", exist_ok=True)

            # Get raw model state dict
            if world_size > 1:
                # FSDP: need to gather full state dict
                from torch.distributed.fsdp import FullStateDictConfig, StateDictType
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                          FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                    state_dict = model.state_dict()
            else:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                state_dict = raw_model.state_dict()

            path = f"checkpoints/{config['run_id']}_step{step}.pt"
            torch.save({
                "model_state_dict": state_dict,
                "config": {k: v for k, v in config.items() if k != "compile"},
                "step": step,
                "run_id": config["run_id"],
                "tokenizer_type": "bpe",
                "best_val_loss": best_val_loss,
            }, path)
            size_mb = os.path.getsize(path) / 1e6
            log(f"  Checkpoint saved: {path} ({size_mb:.0f}MB)")

    # Final
    dt = time.time() - t0

    if is_main:
        log("")
        log("=" * 60)
        log(f"Training complete!")
        log(f"  Total time: {dt/3600:.1f} hours")
        log(f"  Tokens seen: {tokens_seen:,}")
        log(f"  Best val loss: {best_val_loss:.4f}")
        log(f"  Perplexity: {math.exp(best_val_loss):.2f}")
        log(f"  Tokens/sec: {tokens_seen/dt:,.0f}")

        # Save final weights-only checkpoint
        if world_size > 1:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                state_dict = model.state_dict()
        else:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            state_dict = raw_model.state_dict()

        path = f"checkpoints/{config['run_id']}.pt"
        torch.save({
            "model_state_dict": state_dict,
            "config": {k: v for k, v in config.items() if k != "compile"},
            "step": config["max_steps"],
            "run_id": config["run_id"],
            "tokenizer_type": "bpe",
            "best_val_loss": best_val_loss,
        }, path)
        size_mb = os.path.getsize(path) / 1e6
        log(f"  Final checkpoint: {path} ({size_mb:.0f}MB)")
        log("")
        log("  To download:")
        log(f"  scp -P <port> root@<ip>:tinyllm/{path} ./checkpoints/")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
