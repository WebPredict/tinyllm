#!/bin/bash
# Chat with the latest trained model
# Usage: ./scripts/run_chat.sh [optional_checkpoint_path]

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate
python scripts/chat.py "$@"
