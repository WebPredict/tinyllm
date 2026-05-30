"""
Download the tiny Shakespeare dataset for initial pipeline testing.
This is just a sanity check — real training uses React/TypeScript data later.
"""

import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT = Path(__file__).parent.parent / "data" / "input.txt"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists():
        print(f"Already exists: {OUTPUT}")
        print(f"Size: {OUTPUT.stat().st_size:,} bytes")
        return

    print(f"Downloading Shakespeare dataset...")
    print(f"URL: {URL}")

    urllib.request.urlretrieve(URL, OUTPUT)

    size = OUTPUT.stat().st_size
    text = OUTPUT.read_text()
    print(f"Saved to: {OUTPUT}")
    print(f"Size: {size:,} bytes ({len(text):,} characters)")
    print(f"First 200 chars:")
    print(text[:200])


if __name__ == "__main__":
    main()
