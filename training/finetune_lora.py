"""
LoRA fine-tuning for instruction following.

Takes a pretrained model checkpoint and fine-tunes it on instruction pairs
using LoRA (Low-Rank Adaptation) — trains <1% of parameters.

Usage:
  python training/finetune_lora.py --checkpoint checkpoints/cloud_30m_react_bpe.pt
  python training/finetune_lora.py --checkpoint checkpoints/cloud_30m_react_bpe.pt --data data/instruction_pairs.json
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.gpt import GPT


# ─── LoRA Implementation ─────────────────────

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with LoRA adaptation."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # LoRA matrices — these are the only trainable parameters
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Freeze original weights
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

    def forward(self, x):
        # Original output + LoRA adaptation
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out


def apply_lora(model, rank=8, alpha=16, target_modules=None):
    """Apply LoRA to a model's linear layers.

    Args:
        model: The GPT model
        rank: LoRA rank (lower = smaller adapter, less expressive)
        alpha: LoRA scaling factor
        target_modules: Which module names to apply LoRA to.
                       Default: attention Q, K, V, and output projections.
    """
    if target_modules is None:
        target_modules = ["attn.qkv", "attn.proj"]

    lora_params = 0
    frozen_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module's name matches any target
            should_adapt = any(t in name for t in target_modules)

            if should_adapt:
                # Find parent module and attribute name
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = dict(model.named_modules())[parts[0]]
                    attr = parts[1]
                else:
                    parent = model
                    attr = parts[0]

                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, attr, lora_layer)
                lora_params += module.in_features * rank + rank * module.out_features

    # Freeze everything, then unfreeze only LoRA params
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen_params

    print(f"LoRA applied: rank={rank}, alpha={alpha}")
    print(f"  Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
    print(f"  Frozen: {frozen_params:,} ({frozen_params/total*100:.1f}%)")

    return model


# ─── Data Formatting ──────────────────────────

INSTRUCTION_TEMPLATE = """<|instruction|>
{instruction}
<|input|>
{input}
<|output|>
{output}<|endoftext|>"""


def format_pair(pair):
    """Format an instruction pair into the training format."""
    return INSTRUCTION_TEMPLATE.format(
        instruction=pair["instruction"],
        input=pair.get("input", ""),
        output=pair["output"],
    )


def load_instruction_data(data_path, tokenizer):
    """Load and tokenize instruction pairs."""
    with open(data_path) as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} instruction pairs")

    # Tokenize
    tokenized = []
    for pair in pairs:
        text = format_pair(pair)
        encoded = tokenizer.encode(text)
        tokenized.append(torch.tensor(encoded.ids, dtype=torch.long))

    # Stats
    lengths = [len(t) for t in tokenized]
    print(f"  Avg length: {sum(lengths)/len(lengths):.0f} tokens")
    print(f"  Max length: {max(lengths)} tokens")
    print(f"  Min length: {min(lengths)} tokens")

    return tokenized, pairs


# ─── Training ─────────────────────────────────

def train_lora(
    checkpoint_path: str,
    data_path: str = None,
    rank: int = 8,
    alpha: int = 16,
    learning_rate: float = 1e-4,
    epochs: int = 3,
    batch_size: int = 4,
):
    if data_path is None:
        data_path = str(Path(__file__).parent.parent / "data" / "instruction_pairs_seed.json")
        if not Path(data_path).exists():
            # Try the larger generated set
            alt_path = str(Path(__file__).parent.parent / "data" / "instruction_pairs.json")
            if Path(alt_path).exists():
                data_path = alt_path

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    is_bpe = checkpoint.get("tokenizer_type") == "bpe"

    model = GPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model: {model.n_params:,} params")

    # Load tokenizer
    if is_bpe:
        from tokenizers import Tokenizer
        tok_path = Path(__file__).parent.parent / cfg["tokenizer_file"]
        tokenizer = Tokenizer.from_file(str(tok_path))
    else:
        print("Warning: instruction tuning works best with BPE tokenizer")
        return

    # Apply LoRA
    print()
    model = apply_lora(model, rank=rank, alpha=alpha)
    model = model.to(device)

    # Load data
    print()
    tokenized, pairs = load_instruction_data(data_path, tokenizer)

    # Only train LoRA parameters
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    # Training
    print(f"\nTraining LoRA: {epochs} epochs, lr={learning_rate}")
    print("=" * 60)

    block_size = cfg["block_size"]
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        total_loss = 0
        count = 0

        # Shuffle data each epoch
        indices = torch.randperm(len(tokenized))

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]

            # Pad sequences to same length
            batch_tokens = [tokenized[j] for j in batch_indices]
            max_len = min(max(len(t) for t in batch_tokens), block_size)

            # Create padded batch
            x_batch = []
            y_batch = []
            for tokens in batch_tokens:
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                elif len(tokens) < max_len:
                    padding = torch.zeros(max_len - len(tokens), dtype=torch.long)
                    tokens = torch.cat([tokens, padding])

                x_batch.append(tokens[:-1])
                y_batch.append(tokens[1:])

            x = torch.stack(x_batch).to(device)
            y = torch.stack(y_batch).to(device)

            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count
        dt = time.time() - t0

        # Generate a sample
        sample_instruction = pairs[0]["instruction"]
        sample_input = pairs[0].get("input", "")
        prompt = f"<|instruction|>\n{sample_instruction}\n<|input|>\n{sample_input}\n<|output|>\n"
        encoded = tokenizer.encode(prompt)
        idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=100, temperature=0.7, top_k=40)
        generated = tokenizer.decode(output[0].tolist())
        # Extract just the output part
        if "<|output|>" in generated:
            generated = generated.split("<|output|>")[-1]
        if "<|endoftext|>" in generated:
            generated = generated.split("<|endoftext|>")[0]

        print(f"  Epoch {epoch}/{epochs} | loss {avg_loss:.4f} | {dt:.1f}s")
        print(f"    Prompt: {sample_instruction}")
        print(f"    Output: {generated.strip()[:150]}")
        print()

    # Save LoRA checkpoint
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu()

    output_path = Path(checkpoint_path).parent / f"{Path(checkpoint_path).stem}_lora.pt"
    torch.save({
        "lora_state_dict": lora_state,
        "base_checkpoint": str(checkpoint_path),
        "config": cfg,
        "lora_config": {"rank": rank, "alpha": alpha},
        "tokenizer_type": "bpe",
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "data_path": data_path,
            "data_count": len(pairs),
            "final_loss": avg_loss,
        },
    }, str(output_path))

    print("=" * 60)
    print(f"LoRA adapter saved: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1000:.1f} KB")
    print(f"(vs base model: {Path(checkpoint_path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    checkpoint = None
    data = None

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--checkpoint" and i < len(sys.argv) - 1:
            checkpoint = sys.argv[i + 1]
        if arg == "--data" and i < len(sys.argv) - 1:
            data = sys.argv[i + 1]

    if not checkpoint:
        # Try to find latest checkpoint
        cp_dir = Path(__file__).parent.parent / "checkpoints"
        files = sorted(cp_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
        if files:
            checkpoint = str(files[-1])
            print(f"Using latest checkpoint: {checkpoint}")
        else:
            print("Usage: python training/finetune_lora.py --checkpoint <path>")
            sys.exit(1)

    kwargs = {"checkpoint_path": checkpoint}
    if data:
        kwargs["data_path"] = data

    train_lora(**kwargs)
