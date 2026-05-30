"""
Filter the React/TypeScript corpus based on quality analysis.
Produces a cleaned corpus file for comparison training.

Usage: python scripts/filter_data.py [--min-quality 30] [--remove-dupes]
"""

import os
import sys
import json
import hashlib
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})

CORPUS_PATH = Path(__file__).parent.parent / "data" / "react-ts" / "corpus.txt"
ANALYSIS_PATH = Path(__file__).parent.parent / "data" / "react-ts" / "analysis.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "react-ts" / "corpus_filtered.txt"


def main():
    min_quality = 30
    remove_dupes = True

    # Parse args
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--min-quality" and i < len(sys.argv) - 1:
            min_quality = int(sys.argv[i + 1])
        if arg == "--remove-dupes":
            remove_dupes = True

    # Load analysis
    if not ANALYSIS_PATH.exists():
        print("Run analyze_data.py first!")
        sys.exit(1)

    with open(ANALYSIS_PATH) as f:
        analysis = json.load(f)

    quality_map = {item["path"]: item for item in analysis}

    # Parse corpus
    print(f"Reading corpus: {CORPUS_PATH}")
    text = CORPUS_PATH.read_text(encoding="utf-8")

    files = []
    current_path = None
    current_lines = []

    for line in text.split("\n"):
        if line.startswith("// FILE: "):
            if current_path and current_lines:
                files.append({"path": current_path, "content": "\n".join(current_lines)})
            current_path = line[len("// FILE: "):]
            current_lines = []
        else:
            current_lines.append(line)

    if current_path and current_lines:
        files.append({"path": current_path, "content": "\n".join(current_lines)})

    print(f"Parsed {len(files):,} files")

    # Filter
    kept = []
    removed_quality = 0
    removed_dupe = 0
    seen_hashes = set()

    for f in files:
        info = quality_map.get(f["path"], {})
        quality = info.get("quality", 50)

        # Quality filter
        if quality < min_quality:
            removed_quality += 1
            continue

        # Duplicate filter
        if remove_dupes:
            h = hashlib.md5(f["content"].encode()).hexdigest()
            if h in seen_hashes:
                removed_dupe += 1
                continue
            seen_hashes.add(h)

        kept.append(f)

    # Write filtered corpus
    total_chars = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for f in kept:
            header = f"// FILE: {f['path']}\n"
            out.write(header)
            out.write(f["content"])
            out.write("\n\n")
            total_chars += len(header) + len(f["content"]) + 2

    original_chars = sum(len(f["content"]) for f in files)

    print()
    print("=" * 60)
    print("  Filtering Results")
    print("=" * 60)
    print(f"  Original files:     {len(files):,}")
    print(f"  Kept:               {len(kept):,}")
    print(f"  Removed (quality):  {removed_quality:,}")
    print(f"  Removed (dupes):    {removed_dupe:,}")
    print(f"  Original size:      {original_chars/1_000_000:.1f}MB")
    print(f"  Filtered size:      {total_chars/1_000_000:.1f}MB")
    print(f"  Reduction:          {(1 - total_chars/original_chars)*100:.1f}%")
    print(f"  Output:             {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
