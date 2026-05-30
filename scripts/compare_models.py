"""
Compare outputs from multiple models on the same prompts.
Shows side-by-side how different models complete the same code.

Usage:
  python scripts/compare_models.py checkpoint1.pt checkpoint2.pt [checkpoint3.pt ...]
  python scripts/compare_models.py --all   (compares all checkpoints)
"""

import os
import sys
import torch
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.gpt import GPT

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

# Prompts to test — mix of React/TS and general
PROMPTS = [
    "function Button(",
    "import React from",
    "const [count, setCount] = useState(",
    '<div className="',
    "export default function",
    "interface Props {\n",
    "useEffect(() => {\n",
]


def load_model(checkpoint_path):
    """Load a model and return it with its encoding/decoding functions."""
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
    model.eval()

    if is_bpe:
        from tokenizers import Tokenizer
        tok_path = Path(__file__).parent.parent / cfg["tokenizer_file"]
        tokenizer = Tokenizer.from_file(str(tok_path))

        def encode(text):
            return torch.tensor([tokenizer.encode(text).ids], dtype=torch.long)

        def decode(ids):
            return tokenizer.decode(ids)
    else:
        data_path = Path(__file__).parent.parent / cfg["data_file"]
        text = data_path.read_text()
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        def encode(text):
            return torch.tensor([[stoi[c] for c in text if c in stoi]], dtype=torch.long)

        def decode(ids):
            return ''.join([itos.get(i, '?') for i in ids])

    return model, encode, decode, cfg, checkpoint


def generate(model, encode_fn, decode_fn, prompt, max_tokens=150, temperature=0.7, top_k=40):
    """Generate text from a prompt."""
    idx = encode_fn(prompt)
    if idx.shape[1] == 0:
        return "(empty encoding)"
    output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    return decode_fn(output[0].tolist())


def find_final_checkpoints():
    """Find the latest checkpoint for each run_id."""
    if not CHECKPOINT_DIR.exists():
        return []

    # Group by run_id prefix
    runs = {}
    for f in CHECKPOINT_DIR.glob("*.pt"):
        # Extract run_id: everything before _stepNNNN.pt
        name = f.stem
        parts = name.rsplit("_step", 1)
        if len(parts) == 2:
            run_id = parts[0]
            step = int(parts[1])
            if run_id not in runs or step > runs[run_id][1]:
                runs[run_id] = (f, step)

    return [info[0] for info in sorted(runs.values(), key=lambda x: x[0].stat().st_mtime)]


def main():
    # Determine which checkpoints to compare
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        checkpoint_paths = find_final_checkpoints()
    elif len(sys.argv) > 1:
        checkpoint_paths = [Path(p) for p in sys.argv[1:]]
    else:
        print("Usage:")
        print("  python scripts/compare_models.py checkpoint1.pt checkpoint2.pt")
        print("  python scripts/compare_models.py --all")
        sys.exit(1)

    if len(checkpoint_paths) < 2:
        print(f"Need at least 2 checkpoints to compare, found {len(checkpoint_paths)}")
        sys.exit(1)

    # Load all models
    models = []
    for path in checkpoint_paths:
        print(f"Loading: {path.name}...")
        model, encode, decode, cfg, cp = load_model(path)
        run_id = cp.get("run_id", path.stem)
        tok_type = "BPE" if cp.get("tokenizer_type") == "bpe" else "char"
        models.append({
            "name": run_id,
            "model": model,
            "encode": encode,
            "decode": decode,
            "params": model.n_params,
            "tok_type": tok_type,
            "step": cp["step"],
        })

    print()
    print("=" * 80)
    print("  Model Comparison")
    print("=" * 80)

    # Header
    print(f"\n  Models:")
    for m in models:
        print(f"    {m['name']} — {m['params']:,} params, {m['tok_type']}, step {m['step']}")

    # Filter prompts to ones all models can handle
    # (Shakespeare models can't do React prompts meaningfully)
    usable_prompts = []
    for prompt in PROMPTS:
        all_can_encode = True
        for m in models:
            try:
                encoded = m["encode"](prompt)
                if encoded.shape[1] == 0:
                    all_can_encode = False
            except Exception:
                all_can_encode = False
        if all_can_encode:
            usable_prompts.append(prompt)

    if not usable_prompts:
        print("\n  No prompts compatible with all models.")
        print("  Models may be trained on different domains.")
        # Fall back to just showing each model's best
        usable_prompts = PROMPTS

    # Compare on each prompt
    for prompt in usable_prompts:
        print()
        print("─" * 80)
        print(f"  PROMPT: {repr(prompt)}")
        print("─" * 80)

        for m in models:
            try:
                output = generate(m["model"], m["encode"], m["decode"], prompt, max_tokens=150)
                completion = output[len(prompt):] if output.startswith(prompt) else output
                # Clean up for display
                lines = completion.split("\n")
                display = "\n    ".join(lines[:8])
                if len(lines) > 8:
                    display += "\n    ..."
            except Exception as e:
                display = f"(error: {e})"

            print(f"\n  [{m['name']}]")
            print(f"    {display}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
