"""
Train a BPE tokenizer on the React/TypeScript corpus.
Produces a tokenizer that can be used for token-level training.

Usage: python scripts/train_tokenizer.py [--vocab-size 8000]
"""

import os
import sys
import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})

CORPUS_PATH = Path(__file__).parent.parent / "data" / "react-ts" / "corpus.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "react-ts" / "tokenizer"


def train_tokenizer(corpus_path, vocab_size=8000):
    """Train a BPE tokenizer on the corpus."""

    print(f"Training BPE tokenizer")
    print(f"  Corpus: {corpus_path}")
    print(f"  Vocab size: {vocab_size:,}")
    print()

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenizer: split on whitespace and punctuation
    # ByteLevel handles all characters including unicode
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|pad|>", "<|endoftext|>", "<|startoftext|>"],
        show_progress=True,
        min_frequency=2,
    )

    print("Training...")
    tokenizer.train([str(corpus_path)], trainer)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"bpe_{vocab_size}.json"
    tokenizer.save(str(output_path))
    print(f"\nTokenizer saved to: {output_path}")

    return tokenizer, output_path


def analyze_tokenizer(tokenizer, corpus_path, vocab_size):
    """Analyze tokenizer quality."""
    print()
    print("=" * 60)
    print("  Tokenizer Analysis")
    print("=" * 60)

    vocab = tokenizer.get_vocab()
    print(f"\n  Vocabulary size: {len(vocab):,}")

    # Sample some tokens
    # Sort by token ID to see the merge order
    sorted_tokens = sorted(vocab.items(), key=lambda x: x[1])

    # Show some learned tokens
    print(f"\n  First 20 tokens (base characters):")
    for token, idx in sorted_tokens[:20]:
        display = repr(token)
        print(f"    [{idx:>5}] {display}")

    print(f"\n  Tokens 200-230 (early merges):")
    for token, idx in sorted_tokens[200:230]:
        display = repr(token)
        print(f"    [{idx:>5}] {display}")

    print(f"\n  Last 30 tokens (final merges — most common sequences):")
    for token, idx in sorted_tokens[-30:]:
        display = repr(token)
        print(f"    [{idx:>5}] {display}")

    # Test encoding on sample code
    samples = [
        'import React from "react"',
        "const [count, setCount] = useState(0)",
        "function Button({ onClick, children }: ButtonProps) {",
        '<div className="flex items-center gap-2">',
        "export default function Home() {",
        "  return <h1>Hello World</h1>",
        "interface User { id: string; name: string; email: string }",
        "const router = useRouter()",
    ]

    print(f"\n  Sample encodings:")
    total_chars = 0
    total_tokens = 0

    for sample in samples:
        encoded = tokenizer.encode(sample)
        tokens = encoded.tokens
        total_chars += len(sample)
        total_tokens += len(tokens)

        print(f"\n    Input:  {sample}")
        print(f"    Tokens: {tokens}")
        print(f"    Count:  {len(sample)} chars → {len(tokens)} tokens ({len(sample)/len(tokens):.1f} chars/token)")

    avg_compression = total_chars / total_tokens
    print(f"\n  Average compression: {avg_compression:.1f} characters per token")

    # Encode a larger chunk for corpus-level stats
    print(f"\n  Corpus-level stats:")
    text = corpus_path.read_text(encoding="utf-8")
    sample_text = text[:500_000]  # first 500K chars
    encoded = tokenizer.encode(sample_text)
    corpus_compression = len(sample_text) / len(encoded.ids)
    print(f"    500K chars → {len(encoded.ids):,} tokens")
    print(f"    Compression: {corpus_compression:.1f} chars/token")
    print(f"    Effective context at block_size=256: ~{int(256 * corpus_compression)} characters")
    print(f"    vs char-level context: 256 characters")
    print(f"    Context improvement: {corpus_compression:.1f}x")

    # What a 52MB corpus looks like in tokens
    estimated_total_tokens = int(len(text) / corpus_compression)
    print(f"\n    Full corpus: ~{estimated_total_tokens/1_000_000:.1f}M tokens (from {len(text)/1_000_000:.1f}M chars)")

    print()
    print("=" * 60)

    return {
        "vocab_size": len(vocab),
        "compression": corpus_compression,
        "estimated_total_tokens": estimated_total_tokens,
    }


def main():
    vocab_size = 8000
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--vocab-size" and i < len(sys.argv) - 1:
            vocab_size = int(sys.argv[i + 1])

    tokenizer, output_path = train_tokenizer(CORPUS_PATH, vocab_size)
    stats = analyze_tokenizer(tokenizer, CORPUS_PATH, vocab_size)

    # Save stats
    stats_path = OUTPUT_DIR / f"bpe_{vocab_size}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
