"""
Build the RAG index from the React/TS corpus and run test queries.

Usage: python scripts/build_rag_index.py
"""

import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})

sys.path.insert(0, str(Path(__file__).parent.parent))
from hybrid.rag import RAG

CORPUS_PATH = Path(__file__).parent.parent / "data" / "react-ts" / "corpus.txt"


def main():
    print("=" * 60)
    print("  Building RAG Index")
    print("=" * 60)
    print()

    rag = RAG()

    # Build index
    t0 = time.time()
    count = rag.index_corpus(str(CORPUS_PATH))
    dt = time.time() - t0
    print(f"\n  Indexing time: {dt:.1f}s")
    print(f"  Index stats: {rag.stats()}")

    # Test queries
    test_queries = [
        "toggle button component",
        "useState hook with form input",
        "fetch data from API with loading state",
        "TypeScript interface for user props",
        "dark mode theme toggle",
        "responsive navigation sidebar",
        "form validation with error messages",
        "modal dialog component",
    ]

    print()
    print("=" * 60)
    print("  Test Queries")
    print("=" * 60)

    for query in test_queries:
        print(f"\n  Query: \"{query}\"")
        print("  " + "─" * 56)

        results = rag.search(query, top_k=3)
        for i, hit in enumerate(results):
            file = hit["metadata"]["file"]
            distance = hit["distance"]
            preview = hit["text"][:120].replace("\n", " ").strip()
            print(f"  {i+1}. [{distance:.3f}] {file}")
            print(f"     {preview}...")

    # Show an augmented prompt example
    print()
    print("=" * 60)
    print("  Augmented Prompt Example")
    print("=" * 60)
    augmented = rag.augment_prompt("Create a button component with loading state", top_k=3)
    print()
    print(augmented)

    print()
    print("=" * 60)
    print("  RAG index ready! Use hybrid/rag.py in your code:")
    print("    from hybrid.rag import RAG")
    print("    rag = RAG()")
    print("    results = rag.search('your query')")
    print("=" * 60)


if __name__ == "__main__":
    main()
