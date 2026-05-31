#!/bin/bash
# Run the full hybrid chat — model + RAG + validators + memory + cache
# Usage:
#   ./scripts/run_chat_hybrid.sh                              # latest checkpoint
#   ./scripts/run_chat_hybrid.sh checkpoints/some_model.pt    # specific model
#   ./scripts/run_chat_hybrid.sh --verbose                    # show module details

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate
python scripts/chat_hybrid.py "$@"
