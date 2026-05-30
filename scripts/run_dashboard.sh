#!/bin/bash
# Launch the TinyLLM dashboard (API + React frontend)
# Usage: ./scripts/run_dashboard.sh
# Dashboard will be at http://localhost:5173
# API will be at http://localhost:8420

set -e
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════"
echo "  TinyLLM Dashboard"
echo "═══════════════════════════════════════════"
echo ""
echo "  Frontend: http://localhost:5173"
echo "  API:      http://localhost:8420"
echo ""
echo "  Press Ctrl+C to stop both"
echo ""

# Start API server in background
source venv/bin/activate
uvicorn dashboard.api:app --port 8420 --log-level warning &
API_PID=$!

# Start React dev server
cd dashboard
npm run dev &
VITE_PID=$!

# Wait for either to exit, kill both on Ctrl+C
trap "kill $API_PID $VITE_PID 2>/dev/null; exit" INT TERM
wait
