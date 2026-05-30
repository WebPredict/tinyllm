"""
Generate instruction-following training data using Claude API.
Expands the seed pairs into a larger dataset.

Requires: ANTHROPIC_API_KEY environment variable

Usage:
  python scripts/generate_instruction_data.py --count 100
  python scripts/generate_instruction_data.py --count 500 --batch-size 10
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})

SEED_PATH = Path(__file__).parent.parent / "data" / "instruction_pairs_seed.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "instruction_pairs.json"

PROMPT_TEMPLATE = """Generate {count} React/TypeScript instruction-following training examples.

Each example should have:
- "instruction": a clear task description (what to do)
- "input": optional existing code to transform (empty string if creating from scratch)
- "output": the resulting code

Focus on practical daily tasks a frontend developer would do:
- Adding TypeScript types
- Creating React components
- Writing hooks
- Writing tests
- Adding error/loading states
- Refactoring code
- Fixing common bugs
- Adding accessibility
- CSS/Tailwind styling
- Converting between patterns (class→hooks, JS→TS)

Requirements:
- Output must be valid TypeScript/TSX
- Use modern React patterns (functional components, hooks)
- Use Tailwind CSS for styling when relevant
- Keep examples concise (under 30 lines each)
- Vary the difficulty (some simple, some medium)
- Do NOT repeat the seed examples below

Here are seed examples for reference style:
{seeds}

Return ONLY a JSON array of objects. No markdown, no explanation."""


def generate_batch(count: int, seeds: list) -> list:
    """Generate a batch of instruction pairs using Claude API."""
    try:
        import anthropic
    except ImportError:
        print("Install anthropic: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    seed_text = json.dumps(seeds[:5], indent=2)
    prompt = PROMPT_TEMPLATE.format(count=count, seeds=seed_text)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Try to parse JSON from the response
    try:
        # Handle case where response has markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        pairs = json.loads(text)
        return pairs
    except json.JSONDecodeError:
        print(f"  Failed to parse response as JSON")
        print(f"  Response preview: {text[:200]}")
        return []


def main():
    target_count = 100
    batch_size = 10

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--count" and i < len(sys.argv) - 1:
            target_count = int(sys.argv[i + 1])
        if arg == "--batch-size" and i < len(sys.argv) - 1:
            batch_size = int(sys.argv[i + 1])

    # Load seeds
    with open(SEED_PATH) as f:
        seeds = json.load(f)
    print(f"Loaded {len(seeds)} seed examples")

    # Load existing if any
    all_pairs = list(seeds)
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)
        all_pairs = existing
        print(f"Loaded {len(existing)} existing pairs")

    generated = len(all_pairs) - len(seeds)
    remaining = target_count - generated

    if remaining <= 0:
        print(f"Already have {len(all_pairs)} pairs (target: {target_count})")
        return

    print(f"Generating {remaining} more pairs in batches of {batch_size}...")
    print()

    while remaining > 0:
        batch = min(batch_size, remaining)
        print(f"  Generating batch of {batch}... ", end="")

        try:
            new_pairs = generate_batch(batch, seeds)
            if new_pairs:
                all_pairs.extend(new_pairs)
                remaining -= len(new_pairs)
                print(f"got {len(new_pairs)} pairs (total: {len(all_pairs)})")
            else:
                print("failed, retrying...")
                time.sleep(2)
                continue
        except Exception as e:
            print(f"error: {e}")
            time.sleep(5)
            continue

        # Save after each batch
        with open(OUTPUT_PATH, "w") as f:
            json.dump(all_pairs, f, indent=2)

        time.sleep(1)  # rate limit

    print()
    print(f"Done! {len(all_pairs)} total pairs saved to {OUTPUT_PATH}")
    print(f"  Seed pairs: {len(seeds)}")
    print(f"  Generated: {len(all_pairs) - len(seeds)}")


if __name__ == "__main__":
    main()
