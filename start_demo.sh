#!/bin/bash
# ============================================================
#  FYP2 AIOps Demo — One-Command Launcher
#  Usage:  ./start_demo.sh [GEMINI_API_KEY]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate virtual environment ───────────────────────────
source venv/bin/activate

# ── Load .env if present ───────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# ── Optional: override API key as argument ─────────────────
if [ -n "$1" ]; then
    export GEMINI_API_KEY="$1"
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo ""
    echo "⚠️  WARNING: GEMINI_API_KEY is not set."
    echo "   Tier 3 agent will use fallback template reports."
    echo ""
fi

# ── Kill any previous demo processes ──────────────────────
pkill -f "live_pipeline.py" 2>/dev/null || true
pkill -f "streamlit run dashboard" 2>/dev/null || true
sleep 1

# ── Reset dashboard state ──────────────────────────────────
python3 -c "
import json; from pathlib import Path
state = {
    'status':'idle','rows_seen':0,'tier1_flags':0,'tier2_flags':0,
    'alerts':[],'agent_steps':[],'rca_report':None,
    'rf_verdict':None,'lstm_verdict':None,'injected_fault':None
}
Path('demo/state.json').write_text(json.dumps(state))
print('[✓] State reset')
"

# ── Check models exist ────────────────────────────────────
if [ ! -f "data/models/random_forest.pkl" ] || [ ! -f "data/models/lstm_autoencoder.pth" ]; then
    echo ""
    echo "❌ Trained models not found. Running detection pipeline first..."
    echo "   (This takes ~5-10 minutes, please wait)"
    echo ""
    python3 main.py detect
fi
# ── Auto-use Mininet fine-tuned models if available ───────
if [ -f "emulation/mininet_models/random_forest.pkl" ] && [ -f "emulation/mininet_models/lstm_autoencoder.pth" ]; then
    echo "✅ Mininet fine-tuned models found — pipeline will use them automatically"
    echo "   🔬 Demo running on REAL Mininet kernel telemetry models"
else
    echo "   📊 No Mininet models found — using synthetic-trained base models"
fi


echo ""
echo "=============================================="
echo "  🚀 Starting AIOps Live Demo"
echo "=============================================="
echo ""

# ── Start live pipeline in background ────────────────────
echo "[1/2] Starting live pipeline (replay mode)..."
python3 demo/live_pipeline.py --replay > /tmp/live_pipeline.log 2>&1 &
PIPELINE_PID=$!
echo "      PID: $PIPELINE_PID  (logs: /tmp/live_pipeline.log)"

sleep 2  # give it a moment to start

# ── Start Streamlit dashboard ──────────────────────────────
echo "[2/2] Launching Streamlit dashboard..."
echo ""
echo "  🌐 Open your browser at: http://localhost:8501"
echo ""
echo "  💡 Click the fault buttons at the top to inject anomalies!"
echo "  💡 Press Ctrl+C to stop the demo."
echo ""

# Trap Ctrl+C to cleanly stop everything
cleanup() {
    echo ""
    echo "Stopping demo..."
    kill $PIPELINE_PID 2>/dev/null || true
    pkill -f "live_pipeline.py" 2>/dev/null || true
    echo "Demo stopped."
    exit 0
}
trap cleanup INT TERM

streamlit run dashboard/app.py --server.port 8501 --server.headless true
