"""
5M parameter char-level model on Shakespeare.
Purpose: Test scaling from 834K → 5M on laptop.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import config, train

# Override config for 5M model
config.update({
    "n_layer": 6,
    "n_head": 4,
    "n_embd": 256,
    "block_size": 256,
    "dropout": 0.0,

    "batch_size": 64,
    "learning_rate": 5e-4,
    "max_steps": 5000,
    "eval_interval": 250,
    "log_interval": 50,
    "checkpoint_interval": 1000,

    "run_id": "run_5m_shakespeare",
})

if __name__ == "__main__":
    train()
