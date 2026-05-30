#!/bin/bash
# Kill any running training process
# Usage: ./scripts/kill_training.sh

echo "Looking for training processes..."
pids=$(pgrep -f "python.*train.*\.py")

if [ -z "$pids" ]; then
    echo "No training processes found."
else
    echo "Found training processes:"
    ps -p $pids -o pid,etime,command
    echo ""
    echo "Killing..."
    kill $pids
    echo "Done. Processes killed."
    echo "(Latest checkpoint is saved in checkpoints/)"
fi
