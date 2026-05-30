"""
Download and prepare React/TypeScript training data.
Phase 1: Clone high-quality open-source React repos and extract TS/TSX files.
This gives us curated, high-quality code without needing HuggingFace auth.
"""

import os
import subprocess
import shutil
from pathlib import Path

# High-quality React/TypeScript repos (all MIT or similar permissive license)
REPOS = [
    # Component libraries — excellent patterns
    ("shadcn-ui/ui", "shadcn-ui"),
    ("radix-ui/primitives", "radix-primitives"),
    ("mantinedev/mantine", "mantine"),
    ("chakra-ui/chakra-ui", "chakra-ui"),
    ("tremor-so/tremor", "tremor"),

    # Full apps and frameworks — real-world patterns
    ("calcom/cal.com", "calcom"),
    ("vercel/next.js", "nextjs"),
    ("t3-oss/create-t3-app", "create-t3"),
    ("trpc/trpc", "trpc"),
    ("TanStack/query", "tanstack-query"),

    # Utilities and hooks
    ("TanStack/table", "tanstack-table"),
    ("pmndrs/zustand", "zustand"),
    ("react-hook-form/react-hook-form", "react-hook-form"),

    # Smaller high-quality repos
    ("steven-tey/novel", "novel"),
    ("sadmann7/skateshop", "skateshop"),
]

RAW_DIR = Path(__file__).parent.parent / "data" / "react_repos"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "react_ts_corpus.txt"
EXTENSIONS = {".ts", ".tsx", ".jsx"}
SKIP_PATTERNS = [
    "node_modules", "dist", "build", ".next", "coverage",
    "__tests__", "test", "spec", ".d.ts", "package-lock",
    ".min.", "vendor", ".config.", "jest", "cypress",
]


def clone_repo(repo, name, target_dir):
    """Shallow clone a repo (just latest code, no history)."""
    dest = target_dir / name
    if dest.exists():
        print(f"  Already cloned: {name}")
        return dest

    print(f"  Cloning {repo}...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch",
             f"https://github.com/{repo}.git", str(dest)],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        print(f"  Timeout cloning {repo}, skipping")
        if dest.exists():
            shutil.rmtree(dest)
        return None

    # Remove .git to save space
    git_dir = dest / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)

    return dest


def should_skip(filepath):
    """Check if a file should be skipped."""
    path_str = str(filepath).lower()
    return any(pattern in path_str for pattern in SKIP_PATTERNS)


def extract_files(repo_dir, name):
    """Extract TypeScript/TSX files from a cloned repo."""
    files = []
    for ext in EXTENSIONS:
        for filepath in repo_dir.rglob(f"*{ext}"):
            if should_skip(filepath):
                continue
            try:
                content = filepath.read_text(encoding="utf-8")
                # Basic quality filters
                if len(content) < 50:          # too short
                    continue
                if len(content) > 50_000:       # too long (probably generated)
                    continue
                if content.count("\n") < 3:     # basically one-liners
                    continue
                files.append((name, str(filepath.relative_to(repo_dir)), content))
            except (UnicodeDecodeError, OSError):
                continue

    return files


def build_corpus(all_files, output_path):
    """Combine all files into a single training corpus."""
    print(f"\nBuilding corpus from {len(all_files):,} files...")

    total_chars = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for repo_name, filepath, content in all_files:
            # Separator between files so the model can learn file boundaries
            header = f"// FILE: {filepath}\n"
            f.write(header)
            f.write(content)
            f.write("\n\n")
            total_chars += len(header) + len(content) + 2

    print(f"Corpus written to: {output_path}")
    print(f"Total size: {total_chars:,} characters ({total_chars/1_000_000:.1f}MB)")
    return total_chars


def main():
    print("=" * 60)
    print("  Downloading React/TypeScript training data")
    print("=" * 60)
    print()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Clone repos
    print("Step 1: Cloning repos...")
    repo_dirs = []
    for repo, name in REPOS:
        result = clone_repo(repo, name, RAW_DIR)
        if result:
            repo_dirs.append((result, name))
    print()

    # Extract files
    print("Step 2: Extracting TypeScript/TSX files...")
    all_files = []
    for repo_dir, name in repo_dirs:
        files = extract_files(repo_dir, name)
        print(f"  {name}: {len(files)} files")
        all_files.extend(files)
    print(f"\nTotal: {len(all_files):,} files")

    # Build corpus
    print("\nStep 3: Building training corpus...")
    total_chars = build_corpus(all_files, OUTPUT_FILE)

    # Stats
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Repos cloned: {len(repo_dirs)}")
    print(f"  Files extracted: {len(all_files):,}")
    print(f"  Corpus size: {total_chars:,} chars ({total_chars/1_000_000:.1f}MB)")
    print(f"  Output: {OUTPUT_FILE}")
    print()
    print("  Next: train a char model on this with")
    print("  python training/train.py --data data/react_ts_corpus.txt")


if __name__ == "__main__":
    main()
