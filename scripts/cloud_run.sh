#!/bin/bash
# Wrapper that runs any cloud training script with nohup so it survives SSH disconnects.
#
# Usage:
#   bash scripts/cloud_run.sh scripts/cloud_100m_react_bpe.sh
#   bash scripts/cloud_run.sh scripts/cloud_30m_react_bpe.sh
#
# Check progress:
#   tail -30 training_output.log
#
# Watch live:
#   tail -f training_output.log
#
# Check if still running:
#   ps aux | grep python

set -e
cd "$(dirname "$0")/.."

SCRIPT="$1"
LOG="training_output.log"

if [ -z "$SCRIPT" ]; then
    echo "Usage: bash scripts/cloud_run.sh <training_script>"
    echo ""
    echo "Available scripts:"
    ls scripts/cloud_*_react*.sh scripts/cloud_*_shakespeare*.sh 2>/dev/null | sed 's/^/  /'
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "Script not found: $SCRIPT"
    exit 1
fi

echo "═══════════════════════════════════════════"
echo "  Starting training in background"
echo "  Script: $SCRIPT"
echo "  Log:    $LOG"
echo "═══════════════════════════════════════════"
echo ""
echo "  The training will continue even if you disconnect."
echo ""
echo "  Commands:"
echo "    tail -30 $LOG           # check recent progress"
echo "    tail -f $LOG            # watch live"
echo "    ps aux | grep python    # check if still running"
echo "    kill \$(pgrep -f cloud_) # stop training"
echo ""

nohup bash "$SCRIPT" > "$LOG" 2>&1 &
PID=$!

echo "  Started! PID: $PID"
echo ""

# Wait a few seconds and show initial output
sleep 5
echo "  Initial output:"
echo "  ─────────────────────────"
tail -10 "$LOG"
echo ""
echo "  Training is running. You can safely disconnect."
echo "  Check progress with: tail -30 $LOG"
