#!/bin/bash
# Run evals against a model checkpoint
# Usage:
#   ./scripts/run_eval.sh                           ← eval latest checkpoint
#   ./scripts/run_eval.sh checkpoints/some_file.pt  ← eval specific checkpoint
#   ./scripts/run_eval.sh --compare run_id_1 run_id_2  ← compare two runs

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate
python evals/eval.py "$@"
