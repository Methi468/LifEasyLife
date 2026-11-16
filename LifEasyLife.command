#!/bin/zsh
# LifEasyLife — One-Click macOS Launcher
APP_DIR="/Users/methi/.gemini/antigravity/scratch/leica-lif-viewer"

cd "$APP_DIR" || {
  echo "Error: Could not find LifEasyLife directory at $APP_DIR"
  read -k 1 "?Press any key to exit..."
  exit 1
}

source "$APP_DIR/.venv/bin/activate"

clear
echo "========================================================"
echo "  🔬 LifEasyLife — Leica .lif Microscopy Studio"
echo "========================================================"
echo "  Starting LifEasyLife web server..."
echo "  Your browser will open automatically at:"
echo "  http://localhost:8501"
echo ""
echo "  Keep this window open while using LifEasyLife."
echo "  Press Control + C here when you are ready to quit."
echo "========================================================"
echo ""

exec "$APP_DIR/.venv/bin/streamlit" run "$APP_DIR/app.py"
