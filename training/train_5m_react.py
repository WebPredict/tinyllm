"""
5M parameter char-level model on React/TypeScript data.
Purpose: First test of the model on domain-specific code data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train import config, train

# Override config for 5M model on React/TS data
config.update({
    "n_layer": 6,
    "n_head": 4,
    "n_embd": 256,
    "block_size": 256,
    "dropout": 0.1,  # adding dropout since we saw overfitting on Shakespeare

    "batch_size": 64,
    "learning_rate": 5e-4,
    "max_steps": 5000,
    "eval_interval": 250,
    "log_interval": 50,
    "checkpoint_interval": 1000,

    "data_file": "data/react-ts/corpus.txt",
    "train_split": 0.95,  # more data for training since corpus is larger

    "run_id": "run_5m_react_ts",
})

if __name__ == "__main__":
    train()
