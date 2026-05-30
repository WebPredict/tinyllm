"""
Interactive chat with a trained model.
Supports char-level, BPE, and LoRA-adapted models.

Usage:
  python scripts/chat.py                                    ← latest checkpoint
  python scripts/chat.py checkpoints/some_model.pt          ← specific model
  python scripts/chat.py checkpoints/model.pt --lora checkpoints/model_lora.pt
  python scripts/chat.py --instruct                         ← instruction mode
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.gpt import GPT


def find_latest_checkpoint():
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    # Prefer non-LoRA checkpoints
    files = [f for f in checkpoint_dir.glob("*.pt") if "_lora" not in f.name]
    files = sorted(files, key=lambda f: f.stat().st_mtime)
    if not files:
        print("No checkpoints found in checkpoints/")
        sys.exit(1)
    return files[-1]


def find_lora_for(checkpoint_path):
    """Look for a matching LoRA adapter."""
    lora_path = checkpoint_path.parent / f"{checkpoint_path.stem}_lora.pt"
    if lora_path.exists():
        return lora_path
    return None


def load_model(checkpoint_path, lora_path=None):
    print(f"Loading: {checkpoint_path.name}")
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

    # Load LoRA if available
    if lora_path:
        print(f"Loading LoRA: {lora_path.name}")
        from training.finetune_lora import apply_lora
        lora_checkpoint = torch.load(lora_path, map_location="cpu", weights_only=False)
        lora_cfg = lora_checkpoint["lora_config"]
        model = apply_lora(model, rank=lora_cfg["rank"], alpha=lora_cfg["alpha"])
        # Load LoRA weights
        lora_state = lora_checkpoint["lora_state_dict"]
        for name, param in model.named_parameters():
            if name in lora_state:
                param.data = lora_state[name]

    model.eval()

    # Set up encode/decode functions
    if is_bpe:
        from tokenizers import Tokenizer
        tok_path = Path(__file__).parent.parent / cfg["tokenizer_file"]
        tokenizer = Tokenizer.from_file(str(tok_path))

        def encode(text):
            ids = tokenizer.encode(text).ids
            return ids if ids else [0]

        def decode(ids):
            return tokenizer.decode(ids)
    else:
        data_path = Path(__file__).parent.parent / cfg["data_file"]
        text = data_path.read_text()
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        def encode(text):
            return [stoi[c] for c in text if c in stoi] or [0]

        def decode(ids):
            return ''.join([itos.get(i, '?') for i in ids])

    print(f"Model: {model.n_params:,} params")
    print(f"Tokenizer: {'BPE' if is_bpe else 'char'}")
    if lora_path:
        print(f"LoRA: loaded")
    print()

    return model, encode, decode, cfg


def generate(model, encode, decode, prompt, max_tokens=300, temperature=0.8, top_k=40):
    ids = encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    if idx.shape[1] == 0:
        idx = torch.tensor([[0]], dtype=torch.long)
    output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    return decode(output[0].tolist())


def main():
    # Parse args
    checkpoint_path = None
    lora_path = None
    instruct_mode = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--lora" and i + 1 < len(args):
            lora_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--instruct":
            instruct_mode = True
            i += 1
        elif not args[i].startswith("--"):
            checkpoint_path = Path(args[i])
            i += 1
        else:
            i += 1

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    # Auto-detect LoRA
    if lora_path is None:
        lora_path = find_lora_for(checkpoint_path)
        if lora_path:
            print(f"Found LoRA adapter: {lora_path.name}")
            instruct_mode = True  # Auto-enable instruct mode with LoRA

    model, encode, decode, cfg = load_model(checkpoint_path, lora_path)

    print("=" * 50)
    print("  TinyLLM Chat")
    if instruct_mode:
        print("  INSTRUCTION MODE: describe what you want")
        print("  Example: Create a toggle button component")
    else:
        print("  COMPLETION MODE: continue code you type")
        print("  Example: function Button(")
    print()
    print("  Commands: quit, temp=0.5, tokens=500, mode")
    print("=" * 50)
    print()

    temperature = 0.7
    max_tokens = 300

    while True:
        try:
            if instruct_mode:
                prompt = input("Instruction> ")
            else:
                prompt = input("You> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            break
        if prompt.startswith("temp="):
            temperature = float(prompt.split("=")[1])
            print(f"  Temperature set to {temperature}")
            continue
        if prompt.startswith("tokens="):
            max_tokens = int(prompt.split("=")[1])
            print(f"  Max tokens set to {max_tokens}")
            continue
        if prompt.lower() == "mode":
            instruct_mode = not instruct_mode
            mode_name = "INSTRUCTION" if instruct_mode else "COMPLETION"
            print(f"  Switched to {mode_name} mode")
            continue

        # Build prompt based on mode
        if instruct_mode:
            full_prompt = f"<|instruction|>\n{prompt}\n<|input|>\n\n<|output|>\n"
        else:
            full_prompt = prompt

        output = generate(model, encode, decode, full_prompt, max_tokens, temperature)

        # Clean up output for display
        if instruct_mode:
            # Extract just the output part
            if "<|output|>" in output:
                output = output.split("<|output|>")[-1]
            if "<|endoftext|>" in output:
                output = output.split("<|endoftext|>")[0]

        print()
        print(output.strip())
        print()


if __name__ == "__main__":
    main()
