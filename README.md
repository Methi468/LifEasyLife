# 🔬 LifEasy — Leica `.lif` Microscopy Viewer & Real-Time Studio

A Streamlit web application (**LifEasy**) for inspecting Leica Image Format (`.lif`) confocal and widefield fluorescence microscopy files using [`readlif`](https://pypi.org/project/readlif/) and [`tifffile`](https://pypi.org/project/tifffile/).

## Features

- **Leica `.lif` File Parsing (`readlif`)**:
  - Extracts all image **Series**, spatial/temporal **Dimensions** (`X`, `Y`, `Z`, `T`, `M`), **Channels**, **Bit Depths** (8-bit, 12-bit, 16-bit), physical **Scale** (`px/µm`), and XML LUT hints.
- **Z-Stack Navigation & Maximum Intensity Projection (MIP)**:
  - Toggle **Maximum Intensity Projection (MIP)** (or Average/Sum/Minimum projections) across the full Z-stack or a user-selected Z-substack range.
  - Scrub through individual focal slices when MIP is toggled off.
- **Lightroom-Style Independent Per-Channel Adjustments**:
  - **Brightness** (`-100` to `+100`) and **Contrast** (`-100` to `+100`) sliders per channel.
  - **Exposure (EV)** (`-3.0` to `+3.0` stops) and **Gamma ($\gamma$)** (`0.10` to `3.00`).
  - **Black Point / White Point** raw intensity clipping window with one-click **Auto Contrast** (`0.2% – 99.8%` percentile stretch).
  - Live log-scaled **Multi-Channel Intensity Histogram**.
- **Custom Pseudocolors & Fluorophore Presets**:
  - Built-in spectral presets including **Alexa 568 (Orange-Red)**, **Alexa 647 (Magenta)**, **Alexa 647 (Far-Red)**, **Alexa 488 / GFP (Green)**, **Alexa 405 / DAPI (Blue)**, **Alexa 546 / Cy3**, **Alexa 594 / mCherry**, **Cyan**, **Yellow**, and **Grayscale**.
  - Interactive **RGB Hex Color Picker** per channel for arbitrary custom pseudocolors.
- **Export & Overlays**:
  - Optional calibrated **Scale Bar** (`µm`) drawn from `.lif` metadata.
  - Side-by-side **Individual Channel Strip** (in Pseudocolor or High-Contrast Grayscale).
  - Lossless **PNG** and **TIFF** composite image export.

## Quick Start

> **Note:** Large `.lif` files up to **2000 MB (2 GB)** are supported via [.streamlit/config.toml](file:///Users/methi/.gemini/antigravity/scratch/leica-lif-viewer/.streamlit/config.toml) (`server.maxUploadSize = 2000`). Be sure to run `streamlit run app.py` from inside the project directory so Streamlit loads `.streamlit/config.toml` automatically, and restart the Streamlit server if it was already running.

```bash
cd /Users/methi/.gemini/antigravity/scratch/leica-lif-viewer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
