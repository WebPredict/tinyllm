"""
Simple interactive chat with a trained model.
Type a prompt, see what the model generates. Type 'quit' to exit.

Usage: python scripts/chat.py [checkpoint_path]
  If no checkpoint given, uses the latest one in checkpoints/
"""

import sys
import glob
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.gpt import GPT


def find_latest_checkpoint():
    """Find the most recent checkpoint file."""
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    files = sorted(checkpoint_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
    if not files:
        print("No checkpoints found in checkpoints/")
        sys.exit(1)
    return files[-1]


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    # Rebuild character mappings from the data file
    data_path = Path(__file__).parent.parent / cfg["data_file"]
    text = data_path.read_text()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Rebuild model
    model = GPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model: {model.n_params:,} params")
    print(f"Trained for {checkpoint['step']:,} steps")
    print()

    return model, stoi, itos, cfg


def generate(model, stoi, itos, prompt, max_tokens=300, temperature=0.8, top_k=40):
    """Generate text from a prompt."""
    # Encode prompt, skipping unknown characters
    idx = torch.tensor([[stoi[c] for c in prompt if c in stoi]], dtype=torch.long)
    if idx.shape[1] == 0:
        idx = torch.tensor([[stoi['\n']]], dtype=torch.long)

    output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    return ''.join([itos[i] for i in output[0].tolist()])


def main():
    # Load checkpoint
    if len(sys.argv) > 1:
        checkpoint_path = Path(sys.argv[1])
    else:
        checkpoint_path = find_latest_checkpoint()

    model, stoi, itos, cfg = load_model(checkpoint_path)

    print("═" * 50)
    print("  TinyLLM Chat")
    print("  Type a prompt and see what the model generates.")
    print("  Commands: quit, temp=0.5, tokens=500")
    print("═" * 50)
    print()

    temperature = 0.8
    max_tokens = 300

    while True:
        try:
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

        output = generate(model, stoi, itos, prompt, max_tokens, temperature)
        print()
        print(output)
        print()


if __name__ == "__main__":
    main()
