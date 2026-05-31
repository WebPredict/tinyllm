"""
Hybrid chat — full pipeline with RAG, validation, memory, caching, and routing.

This is the end-to-end demo: model + all hybrid modules working together.
Shows which modules fired, latency breakdown, and validation results.

Usage:
  python scripts/chat_hybrid.py                                   # latest checkpoint
  python scripts/chat_hybrid.py checkpoints/cloud_30m_react_bpe.pt  # specific model
  python scripts/chat_hybrid.py --verbose                          # show module details
"""

import os
import sys
import time
import torch
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.gpt import GPT
from hybrid.rag import RAG
from hybrid.memory import StructuredMemory
from hybrid.cache import Cache
from hybrid.router import Router
from hybrid.verifier import VerifierLoop
from hybrid.validators import SyntaxValidator


def find_latest_checkpoint():
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    files = [f for f in checkpoint_dir.glob("*.pt") if "_lora" not in f.name]
    files = sorted(files, key=lambda f: f.stat().st_mtime)
    if not files:
        print("No checkpoints found")
        sys.exit(1)
    return files[-1]


def load_model(checkpoint_path, lora_path=None):
    """Load model with optional LoRA adapter."""
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

    if lora_path and lora_path.exists():
        from training.finetune_lora import apply_lora
        lora_checkpoint = torch.load(lora_path, map_location="cpu", weights_only=False)
        lora_cfg = lora_checkpoint["lora_config"]
        model = apply_lora(model, rank=lora_cfg["rank"], alpha=lora_cfg["alpha"])
        for name, param in model.named_parameters():
            if name in lora_checkpoint["lora_state_dict"]:
                param.data = lora_checkpoint["lora_state_dict"][name]

    model.eval()

    if is_bpe:
        from tokenizers import Tokenizer
        tok_path = Path(__file__).parent.parent / cfg["tokenizer_file"]
        tokenizer = Tokenizer.from_file(str(tok_path))

        def encode(text):
            return tokenizer.encode(text).ids or [0]

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

    return model, encode, decode, cfg


def make_generate_fn(model, encode, decode, block_size):
    """Create a generate function for use in the pipeline."""
    def generate(prompt, max_tokens=300, temperature=0.7, top_k=40):
        ids = encode(prompt)
        # Truncate to block size if needed
        if len(ids) > block_size - 100:
            ids = ids[-(block_size - 100):]
        idx = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        full_text = decode(output[0].tolist())
        # Return only the new text, not the prompt
        prompt_decoded = decode(ids)
        if full_text.startswith(prompt_decoded):
            return full_text[len(prompt_decoded):]
        return full_text
    return generate


def main():
    # Parse args
    checkpoint_path = None
    verbose = False

    for arg in sys.argv[1:]:
        if arg == "--verbose":
            verbose = True
        elif not arg.startswith("--"):
            checkpoint_path = Path(arg)

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    # Only load LoRA if explicitly requested
    lora_path = None
    if "--lora" in sys.argv:
        lora_idx = sys.argv.index("--lora")
        if lora_idx + 1 < len(sys.argv):
            lora_path = Path(sys.argv[lora_idx + 1])
        else:
            auto_lora = checkpoint_path.parent / f"{checkpoint_path.stem}_lora.pt"
            if auto_lora.exists():
                lora_path = auto_lora

    # Load model
    print(f"Loading model: {checkpoint_path.name}")
    if lora_path:
        print(f"Loading LoRA: {lora_path.name}")
    model, encode, decode, cfg = load_model(checkpoint_path, lora_path)
    block_size = cfg["block_size"]
    print(f"Model: {model.n_params:,} params")

    # Initialize hybrid modules
    print("Initializing hybrid modules...")

    rag = RAG()
    rag_count = rag.collection.count()
    print(f"  RAG: {rag_count:,} chunks indexed")

    memory = StructuredMemory()
    mem_stats = memory.stats()
    print(f"  Memory: {mem_stats['total_interactions']} past interactions")

    cache = Cache()
    cache_stats = cache.stats()
    print(f"  Cache: {cache_stats['total_entries']} entries")

    syntax_validator = SyntaxValidator()
    print(f"  Validators: syntax")

    router = Router(cache=cache)
    print(f"  Router: 4-tier routing")

    generate_fn = make_generate_fn(model, encode, decode, block_size)

    print()
    print("=" * 60)
    print("  TinyLLM Hybrid Chat")
    print("  Full pipeline: routing → RAG → generate → validate")
    print()
    print("  Modes:")
    print("    Just type code to complete: function Button(")
    print("    Start with ! for instructions: ! Create a toggle button")
    print()
    print("  Commands: quit, verbose, stats, cache, memory")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input("hybrid> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            break
        if prompt.lower() == "verbose":
            verbose = not verbose
            print(f"  Verbose: {'on' if verbose else 'off'}")
            continue
        if prompt.lower() == "stats":
            print(f"  Memory: {memory.stats()}")
            print(f"  Cache: {cache.stats()}")
            patterns = memory.get_patterns(5)
            if patterns:
                print(f"  Top patterns: {[(p['pattern_name'], p['frequency']) for p in patterns]}")
            continue
        if prompt.lower() == "cache":
            cache.clear_all()
            print("  Cache cleared")
            continue
        if prompt.lower() == "memory":
            recent = memory.get_recent_interactions(5)
            for r in recent:
                print(f"  [{r['task_type']}] success={r['success']}")
            if not recent:
                print("  No interactions recorded yet")
            continue

        t0 = time.time()

        # Instruction mode
        is_instruction = prompt.startswith("!")
        if is_instruction:
            prompt = prompt[1:].strip()
            full_prompt = f"<|instruction|>\n{prompt}\n<|input|>\n\n<|output|>\n"
            task_type = "instruction"
        else:
            full_prompt = prompt
            task_type = "completion"

        # Route
        route = router.route(prompt)
        if verbose:
            print(f"  Route: tier {route.level} — {route.reason}")
            print(f"  Modules: {route.modules}")

        # Cache check
        if route.cache_hit:
            cached = cache.get(prompt)
            if cached:
                dt = (time.time() - t0) * 1000
                print(f"\n{cached}")
                if verbose:
                    print(f"\n  [cache hit, {dt:.0f}ms]")
                print()
                continue

        # RAG augmentation
        rag_context = ""
        if route.level >= 2 and rag_count > 0:
            t_rag = time.time()
            hits = rag.search(prompt, top_k=3)
            rag_context = "\n\n".join(
                f"// Example from: {h['metadata']['file']}\n{h['text'][:300]}"
                for h in hits
            )
            if rag_context:
                full_prompt = f"// Relevant examples:\n{rag_context}\n\n{full_prompt}"
            if verbose:
                dt_rag = (time.time() - t_rag) * 1000
                print(f"  RAG: {len(hits)} results, {dt_rag:.0f}ms")
                for h in hits:
                    print(f"    [{h['distance']:.3f}] {h['metadata']['file']}")

        # Memory context
        mem_context = ""
        if route.level >= 3:
            mem_context = memory.get_context_for_prompt()
            if mem_context and verbose:
                print(f"  Memory: {mem_context[:100]}...")

        # Post-processing helpers
        from hybrid.constrained import BracketBalancer, RepetitionFilter, StopSequence

        def clean_output(text):
            """Apply post-processing to generated text."""
            if is_instruction:
                if "<|output|>" in text:
                    text = text.split("<|output|>")[-1]
                if "<|endoftext|>" in text:
                    text = text.split("<|endoftext|>")[0]
            text = StopSequence(lambda p: p).truncate(text)
            text = RepetitionFilter(lambda p: p).filter_repetition(text)
            text = BracketBalancer(lambda p: p).balance(text)
            return text

        # Generate — with self-consistency if complex query
        use_sc = route.level >= 2 and len(prompt.split()) > 5
        n_candidates = 5 if use_sc else 1

        t_gen = time.time()

        if n_candidates > 1:
            # Self-consistency mode
            print(f"  Generating {n_candidates} candidates...", end="", flush=True)
            candidates = []
            for i in range(n_candidates):
                temp = 0.5 + (0.5 * i / (n_candidates - 1))
                c = generate_fn(full_prompt, temperature=temp)
                c = clean_output(c)
                val = syntax_validator.validate(c)
                candidates.append((c, val))
                symbol = "." if val.passed else "x"
                print(symbol, end="", flush=True)
            print()

            # Pick best
            valid = [(c, v) for c, v in candidates if v.passed]
            if valid:
                # Among valid, pick longest (usually more complete)
                output, val_result = max(valid, key=lambda x: len(x[0]))
            else:
                # None passed — pick one with fewest errors
                output, val_result = min(candidates, key=lambda x: len(x[1].errors))

            attempts = n_candidates
            n_passed = len(valid)
            dt_gen = (time.time() - t_gen) * 1000

            if verbose:
                print(f"  Self-consistency: {n_passed}/{n_candidates} passed validation")
        else:
            # Single generation
            print(f"  Generating...", end="", flush=True)
            output = generate_fn(full_prompt)
            output = clean_output(output)
            print(" done")

            # Validate
            val_result = syntax_validator.validate(output)
            dt_gen = (time.time() - t_gen) * 1000
            attempts = 1
            n_passed = 1 if val_result.passed else 0

            # Verifier loop if validation failed
            if not val_result.passed and route.level >= 2:
                if verbose:
                    print(f"  Validation failed: {val_result.errors[:2]}")
                print(f"  Retrying with error feedback...", end="", flush=True)

                error_prompt = (
                    f"{full_prompt}\n\n"
                    f"// Fix these errors: {'; '.join(val_result.errors[:3])}\n"
                )
                output = generate_fn(error_prompt)
                output = clean_output(output)
                val_result = syntax_validator.validate(output)
                attempts = 2
                print(" done")

        # Display output
        print()
        print(output.strip())
        print()

        # Stats line
        dt_total = (time.time() - t0) * 1000
        status_parts = [f"tier {route.level}"]
        if rag_context:
            status_parts.append("RAG")
        if n_candidates > 1:
            status_parts.append(f"best of {n_candidates} ({n_passed} valid)")
        elif attempts > 1:
            status_parts.append(f"{attempts} attempts")
        if val_result.passed:
            status_parts.append("syntax OK")
        else:
            status_parts.append(f"syntax errors: {len(val_result.errors)}")
        status_parts.append(f"{dt_total:.0f}ms")
        print(f"  [{' | '.join(status_parts)}]")
        print()

        # Record in memory
        memory.record_interaction(
            task_type=task_type,
            input_text=prompt,
            output_text=output[:500],
            success=val_result.passed,
            validator_passed=val_result.passed,
            was_repaired=attempts > 1,
            latency_ms=dt_total,
            modules_used=route.modules,
        )

        # Cache successful results
        if val_result.passed:
            cache.put(prompt, output.strip())


if __name__ == "__main__":
    main()
