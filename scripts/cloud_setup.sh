#!/bin/bash
# Cloud training setup for RunPod (or similar GPU cloud)
#
# RunPod's PyTorch template already has Python, PyTorch, and CUDA.
# This script just installs extra dependencies and downloads data.
#
# Usage:
#   git clone https://github.com/WebPredict/tinyllm.git
#   cd tinyllm
#   bash scripts/cloud_setup.sh

set -e
echo "═══════════════════════════════════════════"
echo "  TinyLLM Cloud Setup"
echo "═══════════════════════════════════════════"
echo ""

# Check for GPU
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "  GPU found!"
else
    echo "  WARNING: No NVIDIA GPU detected."
fi
echo ""

# Check PyTorch + CUDA (should already be installed on RunPod)
echo "Checking PyTorch..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  WARNING: CUDA not available!')
"
echo ""

# Install extra dependencies (tokenizers for BPE)
echo "Installing dependencies..."
pip install tokenizers numpy --quiet 2>&1 | tail -1
echo ""

# Download Shakespeare (tiny, fast)
echo "Downloading Shakespeare data..."
python3 scripts/get_shakespeare.py
echo ""

# Download React/TS data (takes a few minutes — cloning repos)
echo "Downloading React/TypeScript data (this takes 2-3 minutes)..."
python3 scripts/get_react_data.py
echo ""

# Train tokenizer (fast)
echo "Training BPE tokenizer..."
python3 scripts/train_tokenizer.py
echo ""

echo "═══════════════════════════════════════════"
echo "  Setup complete!"
echo "  Run: bash scripts/cloud_test_train.sh"
echo "═══════════════════════════════════════════"
