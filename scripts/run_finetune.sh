#!/bin/bash
# Fine-tune a model with LoRA on instruction pairs
# Usage:
#   ./scripts/run_finetune.sh                                    ← use latest checkpoint + seed data
#   ./scripts/run_finetune.sh --checkpoint checkpoints/model.pt  ← specific checkpoint
#   ./scripts/run_finetune.sh --data data/instruction_pairs.json ← specific data

set -e
cd "$(dirname "$0")/.."
source venv/bin/activate
python training/finetune_lora.py "$@"
