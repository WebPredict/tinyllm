"""
Evaluation harness for TinyLLM models.

Runs a set of eval tasks against a model checkpoint and produces a scorecard.
Results are saved to the SQLite database for comparison across runs.

Usage:
    python evals/eval.py [checkpoint_path]
    python evals/eval.py --compare run_id_1 run_id_2
"""

import sys
import json
import time
import sqlite3
import torch
import math
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.gpt import GPT

# Force unbuffered output
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **{**kwargs, "flush": True})


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def find_latest_checkpoint():
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    files = sorted(checkpoint_dir.glob("*.pt"), key=lambda f: f.stat().st_mtime)
    if not files:
        print("No checkpoints found")
        sys.exit(1)
    return files[-1]


def load_model(checkpoint_path):
    print(f"Loading: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    data_path = Path(__file__).parent.parent / cfg["data_file"]
    text = data_path.read_text()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    model = GPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, stoi, itos, cfg, checkpoint


def generate(model, stoi, itos, prompt, max_tokens=200, temperature=0.8, top_k=40):
    idx = torch.tensor([[stoi[c] for c in prompt if c in stoi]], dtype=torch.long)
    if idx.shape[1] == 0:
        idx = torch.tensor([[stoi.get('\n', 0)]], dtype=torch.long)
    output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    return ''.join([itos[i] for i in output[0].tolist()])


# ─────────────────────────────────────────────
# Eval metrics
# ─────────────────────────────────────────────

def eval_perplexity(model, data, stoi, block_size, num_batches=50, batch_size=32):
    """Calculate perplexity on held-out data. Lower is better."""
    model.eval()
    total_loss = 0
    count = 0

    encoded = torch.tensor([stoi.get(c, 0) for c in data], dtype=torch.long)

    with torch.no_grad():
        for _ in range(num_batches):
            ix = torch.randint(len(encoded) - block_size, (batch_size,))
            x = torch.stack([encoded[i:i+block_size] for i in ix])
            y = torch.stack([encoded[i+1:i+1+block_size] for i in ix])
            _, loss = model(x, y)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def eval_repetition(model, stoi, itos, num_samples=20, length=300):
    """Measure how repetitive the model's output is. Lower is better."""
    total_rep_rate = 0

    for _ in range(num_samples):
        text = generate(model, stoi, itos, "\n", max_tokens=length, temperature=0.8)

        # Check for repeated n-grams
        words = text.split()
        if len(words) < 10:
            total_rep_rate += 1.0
            continue

        # 3-gram repetition rate
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        if not trigrams:
            continue
        unique = len(set(trigrams))
        total = len(trigrams)
        rep_rate = 1.0 - (unique / total)
        total_rep_rate += rep_rate

    return total_rep_rate / num_samples


def eval_structure(model, stoi, itos, num_samples=20, length=300):
    """Check if generated text has expected structural features."""
    results = {
        "has_newlines": 0,
        "has_colons": 0,
        "has_capitals": 0,
        "avg_line_length": 0,
        "valid_chars_ratio": 0,
    }

    for _ in range(num_samples):
        text = generate(model, stoi, itos, "\n", max_tokens=length, temperature=0.8)

        results["has_newlines"] += 1 if "\n" in text else 0
        results["has_colons"] += 1 if ":" in text else 0

        lines = text.split("\n")
        has_cap = any(line and line[0].isupper() for line in lines if line.strip())
        results["has_capitals"] += 1 if has_cap else 0

        line_lengths = [len(line) for line in lines if line.strip()]
        if line_lengths:
            results["avg_line_length"] += sum(line_lengths) / len(line_lengths)

        printable = sum(1 for c in text if c in stoi)
        results["valid_chars_ratio"] += printable / max(len(text), 1)

    # Normalize
    for key in results:
        results[key] /= num_samples

    return results


def eval_prompt_completion(model, stoi, itos, prompts):
    """Test specific prompts and score completions."""
    results = []

    for prompt_text, checks in prompts:
        output = generate(model, stoi, itos, prompt_text, max_tokens=200, temperature=0.5)
        completion = output[len(prompt_text):]

        scores = {}
        for check_name, check_fn in checks.items():
            scores[check_name] = check_fn(completion)

        results.append({
            "prompt": prompt_text,
            "completion": completion[:200],
            "scores": scores,
        })

    return results


# ─────────────────────────────────────────────
# Eval prompt sets
# ─────────────────────────────────────────────

def get_shakespeare_prompts():
    """Prompts for evaluating Shakespeare-trained models."""
    return [
        ("ROMEO:\n", {
            "continues_dialogue": lambda t: ":" in t or "\n" in t,
            "has_words": lambda t: len(t.split()) > 5,
            "no_garbage": lambda t: sum(c.isalpha() or c.isspace() or c in ".,;:!?'-\n" for c in t) / max(len(t),1) > 0.85,
        }),
        ("KING HENRY:\nMy lords, ", {
            "continues_sentence": lambda t: len(t.split()) > 3,
            "has_words": lambda t: len(t.split()) > 5,
            "no_garbage": lambda t: sum(c.isalpha() or c.isspace() or c in ".,;:!?'-\n" for c in t) / max(len(t),1) > 0.85,
        }),
        ("Enter MACBETH.\n\n", {
            "has_character_name": lambda t: any(c.isupper() for c in t[:50]),
            "has_dialogue": lambda t: ":" in t[:100],
            "has_words": lambda t: len(t.split()) > 5,
        }),
        ("What say you, ", {
            "continues": lambda t: len(t.strip()) > 10,
            "has_words": lambda t: len(t.split()) > 3,
            "no_garbage": lambda t: sum(c.isalpha() or c.isspace() or c in ".,;:!?'-\n" for c in t) / max(len(t),1) > 0.85,
        }),
    ]


def _code_char_ratio(t):
    """Check that most characters are valid code characters."""
    valid = sum(c.isalnum() or c.isspace() or c in "{}()[].,;:=<>/'\"-_@!?&|+*#$%^~`\\\n\t" for c in t)
    return valid / max(len(t), 1) > 0.80

def get_react_prompts():
    """Prompts for evaluating React/TypeScript-trained models."""
    return [
        # Component definition
        ("function Button(", {
            "has_braces": lambda t: "{" in t and "}" in t,
            "has_return": lambda t: "return" in t.lower(),
            "no_garbage": _code_char_ratio,
        }),
        # Import statement
        ("import React from", {
            "has_import_or_from": lambda t: "import" in t or "from" in t or "react" in t.lower(),
            "has_line_ending": lambda t: ";" in t or "\n" in t,
            "no_garbage": _code_char_ratio,
        }),
        # Hook usage
        ("const [count, setCount] = useState(", {
            "closes_paren": lambda t: ")" in t,
            "has_code": lambda t: len(t.strip()) > 5,
            "no_garbage": _code_char_ratio,
        }),
        # Export
        ("export default function", {
            "has_function_body": lambda t: "{" in t,
            "has_code": lambda t: len(t.split()) > 3,
            "no_garbage": _code_char_ratio,
        }),
        # JSX div with className
        ('<div className="', {
            "closes_quote": lambda t: '"' in t,
            "has_jsx": lambda t: ">" in t or "/>" in t,
            "no_garbage": _code_char_ratio,
        }),
        # Interface definition
        ("interface Props {\n", {
            "has_type_field": lambda t: ":" in t,
            "closes_brace": lambda t: "}" in t,
            "no_garbage": _code_char_ratio,
        }),
        # useEffect
        ("useEffect(() => {\n", {
            "has_code_body": lambda t: len(t.strip()) > 10,
            "has_closing": lambda t: "}" in t,
            "no_garbage": _code_char_ratio,
        }),
        # Arrow function component
        ("const Card = ({ title, children }: CardProps) => {\n  return (\n", {
            "has_jsx": lambda t: "<" in t,
            "has_closing": lambda t: ")" in t or ">" in t,
            "no_garbage": _code_char_ratio,
        }),
    ]


# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────

def init_eval_db(db_path):
    db = sqlite3.connect(db_path)
    db.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            run_id TEXT,
            checkpoint TEXT,
            timestamp REAL,
            model_params INTEGER,
            eval_type TEXT,
            metric_name TEXT,
            metric_value REAL,
            details TEXT
        )
    """)
    db.commit()
    return db


def save_results(db, run_id, checkpoint_name, model_params, results):
    timestamp = time.time()
    for eval_type, metrics in results.items():
        if isinstance(metrics, dict):
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    db.execute(
                        """INSERT INTO eval_results
                           (run_id, checkpoint, timestamp, model_params, eval_type, metric_name, metric_value)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (run_id, checkpoint_name, timestamp, model_params, eval_type, name, value)
                    )
        elif isinstance(metrics, (int, float)):
            db.execute(
                """INSERT INTO eval_results
                   (run_id, checkpoint, timestamp, model_params, eval_type, metric_name, metric_value)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (run_id, checkpoint_name, timestamp, model_params, eval_type, eval_type, metrics)
            )
    db.commit()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_evals(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
    else:
        checkpoint_path = Path(checkpoint_path)

    model, stoi, itos, cfg, checkpoint = load_model(checkpoint_path)
    run_id = checkpoint.get("run_id", checkpoint_path.stem)

    # Detect domain from data file
    data_file = cfg.get("data_file", "")
    is_react = "react" in data_file.lower() or "corpus" in data_file.lower()
    domain = "react-ts" if is_react else "shakespeare"

    print(f"Model: {model.n_params:,} params")
    print(f"Run ID: {run_id}")
    print(f"Domain: {domain}")
    print(f"Step: {checkpoint['step']:,}")
    print()

    # Get validation data
    data_path = Path(__file__).parent.parent / cfg["data_file"]
    text = data_path.read_text()
    n = int(len(text) * cfg.get("train_split", 0.9))
    val_text = text[n:]

    results = {}

    # 1. Perplexity
    print("Running: perplexity...")
    loss, ppl = eval_perplexity(model, val_text, stoi, cfg["block_size"])
    results["perplexity"] = {"loss": loss, "perplexity": ppl}
    print(f"  Val loss: {loss:.4f}")
    print(f"  Perplexity: {ppl:.2f}")
    print()

    # 2. Repetition
    print("Running: repetition check...")
    rep_rate = eval_repetition(model, stoi, itos)
    results["repetition"] = {"repetition_rate": rep_rate}
    print(f"  Repetition rate: {rep_rate:.4f} ({rep_rate*100:.1f}%)")
    print()

    # 3. Structure
    print("Running: structure check...")
    structure = eval_structure(model, stoi, itos)
    results["structure"] = structure
    print(f"  Has newlines: {structure['has_newlines']*100:.0f}%")
    print(f"  Has colons: {structure['has_colons']*100:.0f}%")
    print(f"  Has capitals: {structure['has_capitals']*100:.0f}%")
    print(f"  Avg line length: {structure['avg_line_length']:.1f}")
    print(f"  Valid chars ratio: {structure['valid_chars_ratio']*100:.1f}%")
    print()

    # 4. Prompt completions
    print("Running: prompt completions...")
    prompts = get_react_prompts() if is_react else get_shakespeare_prompts()
    completions = eval_prompt_completion(model, stoi, itos, prompts)

    prompt_scores = {}
    total_checks = 0
    passed_checks = 0
    for result in completions:
        print(f"  Prompt: {repr(result['prompt'][:40])}")
        print(f"  Output: {repr(result['completion'][:80])}")
        for check, passed in result["scores"].items():
            status = "PASS" if passed else "FAIL"
            print(f"    {check}: {status}")
            total_checks += 1
            if passed:
                passed_checks += 1
        print()

    prompt_pass_rate = passed_checks / max(total_checks, 1)
    results["prompt_completion"] = {"pass_rate": prompt_pass_rate, "total": total_checks, "passed": passed_checks}

    # ─── Summary ─────────────────────────────
    print("=" * 60)
    print("  EVAL SUMMARY")
    print("=" * 60)
    print(f"  Model:            {model.n_params:,} params")
    print(f"  Run:              {run_id}")
    print(f"  Domain:           {domain}")
    print(f"  Val loss:         {loss:.4f}")
    print(f"  Perplexity:       {ppl:.2f}")
    print(f"  Repetition rate:  {rep_rate*100:.1f}%")
    print(f"  Valid chars:      {structure['valid_chars_ratio']*100:.1f}%")
    print(f"  Prompt pass rate: {passed_checks}/{total_checks} ({prompt_pass_rate*100:.0f}%)")
    print("=" * 60)

    # Save to database
    db_path = Path(__file__).parent.parent / "logs" / "training.db"
    db = init_eval_db(str(db_path))
    save_results(db, run_id, checkpoint_path.name, model.n_params, results)
    print(f"\nResults saved to {db_path}")

    return results


def compare_runs(run_id_1, run_id_2):
    """Compare eval results from two runs."""
    db_path = Path(__file__).parent.parent / "logs" / "training.db"
    db = sqlite3.connect(str(db_path))

    print(f"\nComparing: {run_id_1} vs {run_id_2}")
    print("=" * 60)

    rows = db.execute("""
        SELECT eval_type, metric_name, metric_value, run_id
        FROM eval_results
        WHERE run_id IN (?, ?)
        ORDER BY eval_type, metric_name
    """, (run_id_1, run_id_2)).fetchall()

    # Group by metric
    metrics = {}
    for eval_type, name, value, run_id in rows:
        key = f"{eval_type}/{name}"
        if key not in metrics:
            metrics[key] = {}
        metrics[key][run_id] = value

    print(f"\n{'Metric':<35} {'Run 1':>10} {'Run 2':>10} {'Delta':>10}")
    print("─" * 65)
    for key in sorted(metrics):
        v1 = metrics[key].get(run_id_1, None)
        v2 = metrics[key].get(run_id_2, None)
        if v1 is not None and v2 is not None:
            delta = v2 - v1
            sign = "+" if delta > 0 else ""
            print(f"  {key:<33} {v1:>10.4f} {v2:>10.4f} {sign}{delta:>9.4f}")
        elif v1 is not None:
            print(f"  {key:<33} {v1:>10.4f} {'—':>10}")
        elif v2 is not None:
            print(f"  {key:<33} {'—':>10} {v2:>10.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        if len(sys.argv) != 4:
            print("Usage: python evals/eval.py --compare run_id_1 run_id_2")
            sys.exit(1)
        compare_runs(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1:
        run_evals(sys.argv[1])
    else:
        run_evals()
