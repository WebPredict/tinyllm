"""
Analyze the React/TypeScript training corpus for quality issues.
Produces a report on file sizes, duplicates, quality scores, and recommendations.

Usage: python scripts/analyze_data.py [corpus_path]
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})

DEFAULT_CORPUS = Path(__file__).parent.parent / "data" / "react-ts" / "corpus.txt"


def parse_corpus(corpus_path):
    """Parse corpus into individual files (split on // FILE: markers)."""
    text = corpus_path.read_text(encoding="utf-8")
    files = []
    current_path = None
    current_lines = []

    for line in text.split("\n"):
        if line.startswith("// FILE: "):
            if current_path and current_lines:
                files.append({
                    "path": current_path,
                    "content": "\n".join(current_lines),
                })
            current_path = line[len("// FILE: "):]
            current_lines = []
        else:
            current_lines.append(line)

    if current_path and current_lines:
        files.append({
            "path": current_path,
            "content": "\n".join(current_lines),
        })

    return files


def analyze_file(f):
    """Score a single file on multiple quality dimensions."""
    content = f["content"]
    path = f["path"]
    lines = content.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    scores = {}
    issues = []

    # --- Size ---
    char_count = len(content)
    line_count = len(lines)
    scores["char_count"] = char_count
    scores["line_count"] = line_count

    if char_count < 100:
        issues.append("very_short")
    if char_count > 20000:
        issues.append("very_long")

    # --- Content ratio ---
    # What fraction is actual code vs whitespace/comments?
    code_chars = sum(1 for c in content if c.isalnum() or c in "{}()[];:=<>./,!?&|+-*@#$%^~`")
    whitespace_chars = sum(1 for c in content if c.isspace())
    comment_lines = sum(1 for l in lines if l.strip().startswith("//") or l.strip().startswith("*") or l.strip().startswith("/*"))

    code_ratio = code_chars / max(char_count, 1)
    comment_ratio = comment_lines / max(len(non_empty_lines), 1)
    scores["code_ratio"] = code_ratio
    scores["comment_ratio"] = comment_ratio

    if code_ratio < 0.3:
        issues.append("low_code_ratio")
    if comment_ratio > 0.7:
        issues.append("mostly_comments")

    # --- Import heaviness ---
    import_lines = sum(1 for l in lines if l.strip().startswith("import ") or l.strip().startswith("export "))
    import_ratio = import_lines / max(len(non_empty_lines), 1)
    scores["import_ratio"] = import_ratio

    if import_ratio > 0.5 and len(non_empty_lines) < 20:
        issues.append("mostly_imports")

    # --- Repetition ---
    # Check for repeated lines (auto-generated code signature)
    line_counts = Counter(l.strip() for l in lines if l.strip())
    if line_counts:
        most_common_count = line_counts.most_common(1)[0][1]
        unique_ratio = len(line_counts) / max(len(non_empty_lines), 1)
        scores["unique_line_ratio"] = unique_ratio

        if unique_ratio < 0.4:
            issues.append("highly_repetitive")
    else:
        scores["unique_line_ratio"] = 0

    # --- File type signals ---
    ext = Path(path).suffix.lower()
    scores["extension"] = ext

    # Config/setup files
    name_lower = Path(path).name.lower()
    if any(pattern in name_lower for pattern in [
        "config", "setup", "jest", "babel", "webpack", "rollup",
        "eslint", "prettier", "tsconfig", "tailwind.config",
        "next.config", "vite.config", "postcss",
    ]):
        issues.append("config_file")

    # Index/barrel files (just re-exports)
    if name_lower in ("index.ts", "index.tsx", "index.js"):
        export_lines = sum(1 for l in lines if "export" in l)
        if export_lines > 0 and export_lines / max(len(non_empty_lines), 1) > 0.7:
            issues.append("barrel_file")

    # --- React-specific quality ---
    has_jsx = "<" in content and "/>" in content or "</" in content
    has_component = "function" in content or "const" in content
    has_hooks = any(hook in content for hook in ["useState", "useEffect", "useRef", "useMemo", "useCallback"])
    has_types = "interface " in content or "type " in content or ": " in content

    scores["has_jsx"] = has_jsx
    scores["has_component"] = has_component
    scores["has_hooks"] = has_hooks
    scores["has_types"] = has_types

    # --- Overall quality score (0-100) ---
    quality = 50  # baseline

    # Positive signals
    if has_jsx: quality += 10
    if has_hooks: quality += 10
    if has_types: quality += 5
    if 200 < char_count < 10000: quality += 10  # good size
    if code_ratio > 0.5: quality += 5
    if unique_ratio > 0.7 if "unique_line_ratio" in scores else False: quality += 5

    # Negative signals
    if "very_short" in issues: quality -= 20
    if "very_long" in issues: quality -= 10
    if "low_code_ratio" in issues: quality -= 15
    if "mostly_comments" in issues: quality -= 15
    if "mostly_imports" in issues: quality -= 20
    if "highly_repetitive" in issues: quality -= 20
    if "config_file" in issues: quality -= 15
    if "barrel_file" in issues: quality -= 20

    quality = max(0, min(100, quality))
    scores["quality"] = quality

    return {
        **f,
        "scores": scores,
        "issues": issues,
    }


def find_duplicates(files):
    """Find exact and near-duplicate files."""
    # Exact duplicates by content hash
    hashes = defaultdict(list)
    for f in files:
        h = hashlib.md5(f["content"].encode()).hexdigest()
        hashes[h].append(f["path"])

    exact_dupes = {h: paths for h, paths in hashes.items() if len(paths) > 1}

    # Near-duplicates by first 500 chars (catches files that differ only in comments)
    prefix_groups = defaultdict(list)
    for f in files:
        stripped = "".join(f["content"].split())[:500]
        h = hashlib.md5(stripped.encode()).hexdigest()
        prefix_groups[h].append(f["path"])

    near_dupes = {h: paths for h, paths in prefix_groups.items() if len(paths) > 1}

    return exact_dupes, near_dupes


def print_report(files, analyzed, exact_dupes, near_dupes):
    """Print analysis report."""
    print()
    print("=" * 70)
    print("  React/TypeScript Corpus Analysis")
    print("=" * 70)

    # --- Overview ---
    total_chars = sum(f["scores"]["char_count"] for f in analyzed)
    total_lines = sum(f["scores"]["line_count"] for f in analyzed)
    print(f"\n  Total files:      {len(files):,}")
    print(f"  Total characters: {total_chars:,} ({total_chars/1_000_000:.1f}MB)")
    print(f"  Total lines:      {total_lines:,}")

    # --- Size distribution ---
    sizes = [f["scores"]["char_count"] for f in analyzed]
    sizes.sort()
    print(f"\n  File size distribution:")
    print(f"    Min:    {sizes[0]:,} chars")
    print(f"    10th %: {sizes[len(sizes)//10]:,} chars")
    print(f"    Median: {sizes[len(sizes)//2]:,} chars")
    print(f"    90th %: {sizes[len(sizes)*9//10]:,} chars")
    print(f"    Max:    {sizes[-1]:,} chars")

    # --- Extension breakdown ---
    ext_counts = Counter(f["scores"]["extension"] for f in analyzed)
    print(f"\n  File types:")
    for ext, count in ext_counts.most_common():
        pct = count / len(analyzed) * 100
        print(f"    {ext:<8} {count:>6,} ({pct:.1f}%)")

    # --- Quality distribution ---
    qualities = [f["scores"]["quality"] for f in analyzed]
    quality_buckets = Counter()
    for q in qualities:
        if q >= 80: quality_buckets["high (80-100)"] += 1
        elif q >= 50: quality_buckets["medium (50-79)"] += 1
        elif q >= 30: quality_buckets["low (30-49)"] += 1
        else: quality_buckets["very low (0-29)"] += 1

    print(f"\n  Quality distribution:")
    for bucket in ["high (80-100)", "medium (50-79)", "low (30-49)", "very low (0-29)"]:
        count = quality_buckets.get(bucket, 0)
        pct = count / len(analyzed) * 100
        bar = "#" * int(pct / 2)
        print(f"    {bucket:<20} {count:>6,} ({pct:>5.1f}%) {bar}")

    avg_quality = sum(qualities) / len(qualities)
    print(f"    Average quality: {avg_quality:.1f}")

    # --- Issues ---
    issue_counts = Counter()
    for f in analyzed:
        for issue in f["issues"]:
            issue_counts[issue] += 1

    print(f"\n  Common issues:")
    for issue, count in issue_counts.most_common():
        pct = count / len(analyzed) * 100
        print(f"    {issue:<25} {count:>6,} ({pct:.1f}%)")

    # --- Duplicates ---
    exact_dupe_files = sum(len(paths) - 1 for paths in exact_dupes.values())
    near_dupe_files = sum(len(paths) - 1 for paths in near_dupes.values())
    print(f"\n  Duplicates:")
    print(f"    Exact duplicates:  {exact_dupe_files:,} files ({len(exact_dupes)} groups)")
    print(f"    Near-duplicates:   {near_dupe_files:,} files ({len(near_dupes)} groups)")

    # --- React/TS features ---
    jsx_count = sum(1 for f in analyzed if f["scores"]["has_jsx"])
    hooks_count = sum(1 for f in analyzed if f["scores"]["has_hooks"])
    types_count = sum(1 for f in analyzed if f["scores"]["has_types"])
    print(f"\n  React/TypeScript features:")
    print(f"    Has JSX:        {jsx_count:>6,} ({jsx_count/len(analyzed)*100:.1f}%)")
    print(f"    Has hooks:      {hooks_count:>6,} ({hooks_count/len(analyzed)*100:.1f}%)")
    print(f"    Has types:      {types_count:>6,} ({types_count/len(analyzed)*100:.1f}%)")

    # --- Filtering recommendations ---
    low_quality = [f for f in analyzed if f["scores"]["quality"] < 30]
    removable = len(low_quality) + exact_dupe_files
    removable_chars = sum(f["scores"]["char_count"] for f in low_quality)

    print(f"\n  Recommendations:")
    print(f"    Files to remove (quality < 30 + exact dupes): {removable:,}")
    print(f"    Characters to remove: {removable_chars:,} ({removable_chars/total_chars*100:.1f}%)")
    print(f"    Estimated corpus after filtering: {(total_chars - removable_chars)/1_000_000:.1f}MB")

    # --- Examples of low quality files ---
    print(f"\n  Sample low-quality files:")
    for f in sorted(low_quality, key=lambda x: x["scores"]["quality"])[:10]:
        print(f"    [{f['scores']['quality']:>3}] {f['path'][:70]}  issues: {', '.join(f['issues'])}")

    # --- Examples of high quality files ---
    high_quality = [f for f in analyzed if f["scores"]["quality"] >= 80]
    print(f"\n  Sample high-quality files:")
    for f in sorted(high_quality, key=lambda x: -x["scores"]["quality"])[:10]:
        print(f"    [{f['scores']['quality']:>3}] {f['path'][:70]}")

    print()
    print("=" * 70)

    return {
        "total_files": len(files),
        "total_chars": total_chars,
        "avg_quality": avg_quality,
        "exact_dupes": exact_dupe_files,
        "near_dupes": near_dupe_files,
        "low_quality_files": len(low_quality),
        "removable_chars": removable_chars,
    }


def main():
    corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORPUS

    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        sys.exit(1)

    print(f"Analyzing: {corpus_path}")
    print(f"Size: {corpus_path.stat().st_size / 1_000_000:.1f}MB")

    print("\nParsing corpus into files...")
    files = parse_corpus(corpus_path)
    print(f"Found {len(files):,} files")

    print("\nAnalyzing file quality...")
    analyzed = [analyze_file(f) for f in files]

    print("\nFinding duplicates...")
    exact_dupes, near_dupes = find_duplicates(files)

    stats = print_report(files, analyzed, exact_dupes, near_dupes)

    # Save detailed results for the filter script
    output_path = corpus_path.parent / "analysis.json"
    results = []
    for f in analyzed:
        results.append({
            "path": f["path"],
            "quality": f["scores"]["quality"],
            "issues": f["issues"],
            "char_count": f["scores"]["char_count"],
            "has_jsx": f["scores"]["has_jsx"],
            "has_hooks": f["scores"]["has_hooks"],
            "has_types": f["scores"]["has_types"],
        })
    with open(output_path, "w") as out:
        json.dump(results, out, indent=2)
    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
