# TinyLLM

Building a small language model from scratch with a hybrid architecture that makes it punch above its weight. Focused on React/TypeScript code generation.

## What is this?

A hands-on project to train a small LLM (targeting 1B-3B parameters) and wrap it in a hybrid system of retrieval, validators, tools, and memory — making it genuinely useful for narrow-domain tasks despite its small size.

The core idea:

```
Small LLM
+ RAG (retrieval-augmented generation)
+ structured memory
+ validators (TypeScript compiler, ESLint, test runner)
+ tools (code execution, calculators)
+ constrained decoding
+ self-improvement loop
= useful React/TypeScript development assistant
```

## Current status

Phase 1: Pipeline validation — training small models on laptop, building eval harness and dashboard.

See [PROGRESS.md](PROGRESS.md) for detailed status.

## Project structure

```
model/gpt.py               GPT implementation from scratch
training/train.py           Character-level training loop
training/train_bpe.py       Token-level (BPE) training loop
evals/eval.py               Evaluation harness
dashboard/                  React/TypeScript monitoring dashboard
scripts/                    Run scripts for all operations
data/                       Training data (not in git, see below)
checkpoints/                Saved model weights (not in git)
logs/training.db            Training and eval metrics
```

## Getting started

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download data
python scripts/get_shakespeare.py          # Shakespeare (sanity check)
python scripts/get_react_data.py           # React/TS corpus (52MB from 15 repos)

# Train tokenizer
python scripts/train_tokenizer.py

# Train a model
./scripts/run_first_train.sh               # 834K char model (13 min)
./scripts/run_5m_react.sh                  # 5M char model on React/TS
./scripts/run_5m_react_bpe.sh              # 5M BPE model on React/TS

# Evaluate
./scripts/run_eval.sh                      # Eval latest checkpoint
./scripts/run_eval.sh --compare run1 run2  # Compare two runs

# Chat with a model
./scripts/run_chat.sh

# Dashboard
./scripts/run_dashboard.sh                 # http://localhost:5173

# Other tools
python scripts/analyze_data.py             # Data quality analysis
python scripts/filter_data.py              # Filter corpus by quality
./scripts/kill_training.sh                 # Kill running training
```

## Key results so far

| Run | Params | Data | Val Loss | Perplexity | Overfitting |
|---|---|---|---|---|---|
| 834K char Shakespeare | 834K | 1.1MB | 1.54 | 4.91 | Mild |
| 5M char Shakespeare | 4.8M | 1.1MB | 1.54 | 5.00 | Severe |
| 5M char React/TS | 5.2M | 52MB | 0.66 | 1.95 | None |

Key learnings:
- More data matters more than bigger models (Shakespeare 5M overfits, React/TS 5M doesn't)
- A 5M char model generates plausible React/JSX code (valid component structure, Tailwind classes, TypeScript types)
- BPE tokenizer gives 3.2x compression → 3.2x more context per token

## Plan

See [tiny_llm_hybrid_implementation_plan.md](tiny_llm_hybrid_implementation_plan.md) for the full implementation plan covering laptop experiments through cloud training to hybrid architecture.
