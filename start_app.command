#!/bin/zsh
cd "/Users/methi/.gemini/antigravity/scratch/leica-lif-viewer" || exit 1
source .venv/bin/activate
streamlit run app.py
