"""
Real-time HTML5 Canvas Studio for Leica .lif Microscopy Images.

Strictly separates:
1. 'Save for Analysis' (Raw Data):
   - Uses isolated raw integer arrays (`uint8` or `uint16`) directly from `readlif`.
   - Completely bypasses all visual LUTs, brightness, contrast, gamma, clipping,
     pseudocolors, and scale bar overlays.
   - Exports unmodified scientific `minisblack` TIFF files (16-bit or 8-bit).
2. 'Save for Presentation' (Visual Render):
   - Exports the 8-bit RGB visual image exactly as it currently looks on the screen
     (with all Lightroom-style tonal adjustments, clipping, pseudocolors, and scale bar baked in).
   - Includes a dropdown menu to select the output format: PNG, JPEG, or TIFF (8-bit RGB).
"""

import base64
import io
import json
from typing import Any, Dict, List, Optional
import numpy as np
import tifffile

from image_processing import DEFAULT_CHANNEL_PRESETS, FLUOROPHORE_PRESETS
from lif_loader import SeriesMetadata, map_lut_hint_to_preset


def _build_tifffile_imagej_templates(
    num_channels: int,
    height: int,
    width: int,
    scale_x: Optional[float],
    scale_y: Optional[float],
) -> Dict[str, Dict[str, str]]:
    """
    Use `tifffile.imwrite(..., imagej=True)` to generate exact ImageJ hyperstack
    binary prefix and suffix templates for any number of selected channels `k` (1..num_channels)
    and bit depth (`8` or `16`).

    When assembling a raw analysis TIFF in the studio for `k` selected channels,
    concatenating `[prefix, ch_0_bytes, ..., ch_{k-1}_bytes, suffix]` produces a file
    that is bit-for-bit identical to calling `tifffile.imwrite(buf, np.stack(channels, axis=0), imagej=True)`
    with shape `(C, Y, X)` (`axes='CYX'`).
    """
    templates: Dict[str, Dict[str, str]] = {}
    resolution = None
    if scale_x and scale_x > 0:
        sy = scale_y if (scale_y and scale_y > 0) else scale_x
        resolution = (float(scale_x), float(sy))

    for bits, dt in ((8, np.uint8), (16, np.uint16)):
        bytes_per_px = bits // 8
        for k in range(1, num_channels + 1):
            shape = (height, width) if k == 1 else (k, height, width)
            axes = "YX" if k == 1 else "CYX"
            dummy = np.zeros(shape, dtype=dt)
            meta_dict: Dict[str, Any] = {"axes": axes, "mode": "composite"}
            if resolution is not None:
                meta_dict["unit"] = "um"

            buf = io.BytesIO()
            if resolution is not None:
                tifffile.imwrite(
                    buf,
                    dummy,
                    imagej=True,
                    photometric="minisblack",
                    resolution=resolution,
                    metadata=meta_dict,
                )
            else:
                tifffile.imwrite(
                    buf,
                    dummy,
                    imagej=True,
                    photometric="minisblack",
                    metadata=meta_dict,
                )
            full_bytes = buf.getvalue()
            with tifffile.TiffFile(io.BytesIO(full_bytes)) as tf:
                offset0 = int(tf.pages[0].dataoffsets[0])
            raw_byte_len = k * height * width * bytes_per_px
            prefix_b64 = base64.b64encode(full_bytes[:offset0]).decode("ascii")
            suffix_b64 = base64.b64encode(full_bytes[offset0 + raw_byte_len :]).decode("ascii")
            templates[f"{bits}_{k}"] = {
                "prefixB64": prefix_b64,
                "suffixB64": suffix_b64,
            }
    return templates


def _compute_objective_calibrations(meta: SeriesMetadata, width: int) -> Dict[str, Any]:
    """
    Compute accurate pixels-per-micrometer (px/µm) calibrations for 10x, 40x, and 60x
    objective lenses using the Leica .lif series metadata.
    """
    zoom = meta.scanner_zoom if (meta.scanner_zoom and meta.scanner_zoom > 0) else 1.0
    lif_scale = float(meta.scale_x) if (meta.scale_x and meta.scale_x > 0) else None
    lif_mag = (
        float(meta.objective_magnification)
        if (meta.objective_magnification and meta.objective_magnification > 0)
        else None
    )

    if lif_scale is not None:
        if lif_mag is not None:
            base_per_1x = lif_scale / lif_mag
        else:
            fov_um = max((width - 1) / lif_scale, 1e-3)
            est_mag = 11000.0 / (fov_um * zoom)
            standard_mags = [5.0, 10.0, 20.0, 40.0, 60.0, 63.0, 100.0]
            lif_mag = min(standard_mags, key=lambda m: abs(m - est_mag))
            base_per_1x = lif_scale / lif_mag

        cal_10x = round(base_per_1x * 10.0, 4)
        cal_40x = round(base_per_1x * 40.0, 4)
        cal_60x = round(base_per_1x * 60.0, 4)
    else:
        eff_w = max(width - 1, 1)
        cal_10x = round(eff_w / (11000.0 / (10.0 * zoom)), 4)
        cal_40x = round(eff_w / (11000.0 / (40.0 * zoom)), 4)
        cal_60x = round(eff_w / (11000.0 / (60.0 * zoom)), 4)

    if lif_mag is not None:
        if lif_mag < 25.0:
            default_obj = "10x"
        elif lif_mag < 50.0:
            default_obj = "40x"
        else:
            default_obj = "60x"
    else:
        default_obj = "40x"

    obj_info_parts = []
    if meta.objective_name:
        obj_info_parts.append(meta.objective_name)
    elif lif_mag:
        obj_info_parts.append(f"{int(lif_mag)}x Objective")
    if zoom and abs(zoom - 1.0) > 0.01:
        obj_info_parts.append(f"Zoom {zoom:.2f}×")
    if lif_scale:
        fov_x = (width - 1) / lif_scale
        obj_info_parts.append(f"FOV {fov_x:.1f} µm ({lif_scale:.3f} px/µm)")

    detected_summary = " • ".join(obj_info_parts) if obj_info_parts else "Standard Leica Calibration"

    return {
        "defaultObjective": default_obj,
        "detectedSummary": detected_summary,
        "lifScalePxPerUm": round(lif_scale, 4) if lif_scale else None,
        "lifMagnification": lif_mag,
        "scannerZoom": round(zoom, 3),
        "objectiveDefaults": {
            "10x": cal_10x,
            "40x": cal_40x,
            "60x": cal_60x,
        },
    }


def build_realtime_studio_html(
    dataset_id: str,
    meta: SeriesMetadata,
    presentation_planes_2d: List[np.ndarray],
    mode_label: str,
    raw_analysis_planes_2d: Optional[List[np.ndarray]] = None,
) -> str:
    """
    Build the self-contained HTML/JS/CSS for the real-time microscopy studio.

    Parameters
    ----------
    dataset_id : str
        Unique identifier for caching UI state per file/series.
    meta : SeriesMetadata
        Metadata for the selected series.
    presentation_planes_2d : List[np.ndarray]
        2D float32 arrays used exclusively for visual presentation rendering.
    mode_label : str
        Human-readable label of current Z-slice or MIP state.
    raw_analysis_planes_2d : Optional[List[np.ndarray]]
        Strictly isolated, unmodified raw integer (`uint8` or `uint16`) 2D arrays
        directly from `readlif` used exclusively for 'Save for Analysis (Raw Data)'.
    """
    num_channels = len(presentation_planes_2d)
    if num_channels == 0:
        return "<div style='color:white;'>No channels available in this series.</div>"

    if raw_analysis_planes_2d is None:
        raw_analysis_planes_2d = presentation_planes_2d

    height, width = presentation_planes_2d[0].shape

    channels_payload: List[Dict[str, Any]] = []
    for c_idx in range(num_channels):
        vis_plane = presentation_planes_2d[c_idx]
        raw_plane = raw_analysis_planes_2d[c_idx]

        p_min = float(np.min(vis_plane))
        p_max = float(np.max(vis_plane))
        p2, p998 = np.percentile(vis_plane, [0.2, 99.8]) if vis_plane.size > 0 else (0.0, 255.0)
        if p998 <= p2:
            p2, p998 = p_min, max(p_max, p_min + 1.0)

        b_depth = meta.bit_depths[c_idx] if c_idx < len(meta.bit_depths) else 8
        theoretical_max = float((1 << b_depth) - 1) if b_depth in (8, 10, 12, 16) else p_max
        slider_max = max(p_max, min(theoretical_max, max(p_max * 1.25, 255.0)), 1.0)

        # 1. Visual presentation plane (mapped to uint16 LUT indices for 60 FPS slider rendering)
        if p_min >= 0.0 and slider_max <= 65535.0:
            scale_to_u16 = 1.0
            offset_to_u16 = 0.0
            u16_vis_arr = np.clip(np.rint(vis_plane), 0, 65535).astype("<u2")
            u16_slider_max = int(min(65535, max(1, round(slider_max))))
        else:
            offset_to_u16 = p_min
            span = max(slider_max - p_min, 1e-6)
            scale_to_u16 = 65535.0 / span
            u16_vis_arr = np.clip(
                np.rint((vis_plane - offset_to_u16) * scale_to_u16), 0, 65535
            ).astype("<u2")
            u16_slider_max = 65535

        b64_vis_u16 = base64.b64encode(u16_vis_arr.tobytes()).decode("ascii")

        # 2. Strictly isolated Raw Analysis buffer (exact unmodified uint8 or uint16 from readlif)
        raw_bits = 8 if b_depth <= 8 and raw_plane.dtype == np.uint8 else 16
        if raw_bits == 8:
            raw_exact_arr = np.asarray(raw_plane, dtype=np.uint8)
        else:
            raw_exact_arr = np.asarray(raw_plane, dtype="<u2")
        b64_raw_exact = base64.b64encode(raw_exact_arr.tobytes()).decode("ascii")

        lut_hint = meta.channel_luts[c_idx] if c_idx < len(meta.channel_luts) else ""
        default_preset = map_lut_hint_to_preset(lut_hint, c_idx)
        if default_preset not in FLUOROPHORE_PRESETS:
            default_preset = DEFAULT_CHANNEL_PRESETS[c_idx % len(DEFAULT_CHANNEL_PRESETS)]
        default_hex = FLUOROPHORE_PRESETS[default_preset].hex_color

        ch_name = (
            meta.channel_names[c_idx]
            if c_idx < len(meta.channel_names)
            else f"Channel {c_idx + 1}"
        )

        channels_payload.append(
            {
                "index": c_idx,
                "name": ch_name,
                "bitDepth": b_depth,
                "rawContainerBits": raw_bits,
                "rawMin": int(np.min(raw_exact_arr)),
                "rawMax": int(np.max(raw_exact_arr)),
                "autoLow": round(float(p2), 2),
                "autoHigh": round(float(p998), 2),
                "sliderMax": round(slider_max, 2),
                "u16SliderMax": u16_slider_max,
                "scaleToU16": scale_to_u16,
                "offsetToU16": offset_to_u16,
                "defaultPreset": default_preset,
                "defaultHex": default_hex,
                "b64U16": b64_vis_u16,
                "b64RawExact": b64_raw_exact,
            }
        )

    presets_payload = {
        name: {
            "hex": p.hex_color,
            "ex": p.ex_nm,
            "em": p.em_nm,
            "desc": p.description,
        }
        for name, p in FLUOROPHORE_PRESETS.items()
    }

    cal_info = _compute_objective_calibrations(meta, width)
    imagej_templates = _build_tifffile_imagej_templates(
        num_channels=num_channels,
        height=height,
        width=width,
        scale_x=meta.scale_x,
        scale_y=meta.scale_y,
    )

    config_json = json.dumps(
        {
            "storageKey": f"lif_studio_v4_{dataset_id}_s{meta.index}",
            "seriesName": meta.name,
            "modeLabel": mode_label,
            "width": width,
            "height": height,
            "numChannels": num_channels,
            **cal_info,
            "imagejTemplates": imagej_templates,
            "presets": presets_payload,
            "channels": channels_payload,
        }
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<style>
  :root {{
    --bg-dark: #0e1117;
    --bg-panel: #161b22;
    --bg-card: #1f2630;
    --border-col: #30363d;
    --text-main: #f0f6fc;
    --text-muted: #8b949e;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
  }}
  * {{
    box-sizing: border-box;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  body {{
    margin: 0;
    padding: 8px;
    background: var(--bg-dark);
    color: var(--text-main);
    overflow-x: hidden;
  }}
  .studio-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1fr) 425px;
    gap: 16px;
    align-items: start;
  }}
  .studio-grid.cinema-mode {{
    grid-template-columns: 1fr;
  }}
  @media (max-width: 1100px) {{
    .studio-grid {{
      grid-template-columns: 1fr;
    }}
  }}
  /* Left Viewport Panel */
  .viewport-panel {{
    background: var(--bg-panel);
    border: 1px solid var(--border-col);
    border-radius: 10px;
    padding: 12px;
  }}
  .viewport-toolbar {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    padding-bottom: 10px;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--border-col);
  }}
  .meta-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
  }}
  .badge {{
    background: var(--bg-card);
    border: 1px solid var(--border-col);
    padding: 4px 9px;
    border-radius: 6px;
    font-size: 12px;
    color: var(--text-main);
  }}
  .badge-highlight {{
    background: rgba(59, 130, 246, 0.15);
    border-color: rgba(59, 130, 246, 0.5);
    color: #93c5fd;
    font-weight: 600;
  }}
  .zoom-controls {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
  }}
  .canvas-stage {{
    width: 100%;
    min-height: 560px;
    max-height: 84vh;
    overflow: auto;
    background: #05070a;
    border: 1px solid #22272e;
    border-radius: 8px;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    position: relative;
  }}
  #main-canvas {{
    display: block;
    background: #000000;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.65);
    image-rendering: auto;
  }}
  #main-canvas.pixelated {{
    image-rendering: pixelated;
  }}
  /* Individual channels strip below main viewport */
  .strip-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 16px;
    margin-bottom: 8px;
  }}
  .strip-title {{
    font-size: 14px;
    font-weight: 600;
  }}
  .channel-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 10px;
  }}
  .ch-thumb-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-col);
    border-radius: 8px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .ch-thumb-card.muted {{
    opacity: 0.5;
  }}
  .ch-thumb-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    font-weight: 600;
  }}
  .ch-thumb-canvas {{
    width: 100%;
    height: auto;
    background: #000;
    border-radius: 4px;
    border: 1px solid #222;
  }}
  /* Right Studio Dock & Collapsible Dropdown Panels */
  .dock-panel {{
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}
  .panel-box {{
    background: var(--bg-panel);
    border: 1px solid var(--border-col);
    border-radius: 10px;
    overflow: hidden;
  }}
  .panel-header {{
    padding: 11px 13px;
    font-size: 14px;
    font-weight: 700;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
    background: rgba(255, 255, 255, 0.02);
    transition: background 0.15s ease;
  }}
  .panel-header:hover {{
    background: rgba(255, 255, 255, 0.06);
  }}
  .panel-header-left {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .chevron {{
    display: inline-block;
    font-size: 11px;
    color: var(--text-muted);
    transition: transform 0.18s ease;
  }}
  .panel-box.collapsed .chevron {{
    transform: rotate(-90deg);
  }}
  .panel-body {{
    padding: 12px;
    border-top: 1px solid var(--border-col);
  }}
  .panel-box.collapsed .panel-body {{
    display: none;
  }}
  /* Sub-sections for Export (Analysis vs Presentation) */
  .export-subbox {{
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-col);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
  }}
  .export-subbox-analysis {{
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.05);
  }}
  .export-subbox-pres {{
    border-color: rgba(59, 130, 246, 0.45);
    background: rgba(59, 130, 246, 0.05);
  }}
  .subbox-title {{
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .subbox-desc {{
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 8px;
    line-height: 1.35;
  }}
  /* Buttons */
  .btn {{
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border-col);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.12s ease;
  }}
  .btn:hover:not(:disabled) {{
    background: #2b3441;
    border-color: #485260;
  }}
  .btn:disabled {{
    opacity: 0.4;
    cursor: not-allowed;
  }}
  .btn.active {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
  }}
  .btn-primary {{
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
    padding: 8px 12px;
    font-size: 13px;
  }}
  .btn-primary:hover {{
    background: var(--accent-hover);
  }}
  .btn-success {{
    background: #10b981;
    border-color: #059669;
    color: #fff;
    font-weight: 600;
    padding: 8px 12px;
    font-size: 13px;
  }}
  .btn-success:hover {{
    background: #059669;
  }}
  .btn-row {{
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }}
  /* Merge Checklist */
  .merge-list {{
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-bottom: 8px;
  }}
  .merge-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-card);
    border: 1px solid var(--border-col);
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
  }}
  .merge-item-left {{
    display: flex;
    align-items: center;
    gap: 7px;
    cursor: pointer;
  }}
  .color-dot {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
    display: inline-block;
    border: 1px solid rgba(255,255,255,0.4);
  }}
  /* Scale Bar Controls */
  .form-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 12.5px;
  }}
  .num-input, .text-input, .select-input {{
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border-col);
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12.5px;
  }}
  .num-input {{
    width: 95px;
    text-align: right;
  }}
  .text-input {{
    width: 100%;
  }}
  .select-input {{
    width: 100%;
  }}
  .cal-info-box {{
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 6px;
    padding: 7px 9px;
    font-size: 11.5px;
    color: #bfdbfe;
    margin-bottom: 9px;
    line-height: 1.45;
  }}
  /* Channel Adjustment Card (also collapsible) */
  .ch-card {{
    background: var(--bg-card);
    border: 1px solid var(--border-col);
    border-radius: 8px;
    margin-bottom: 10px;
    overflow: hidden;
  }}
  .ch-card-hdr {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    background: rgba(255, 255, 255, 0.02);
    cursor: pointer;
    user-select: none;
  }}
  .ch-card-hdr:hover {{
    background: rgba(255, 255, 255, 0.05);
  }}
  .ch-card-title {{
    font-size: 13px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 7px;
  }}
  .ch-card-body {{
    padding: 10px;
    border-top: 1px solid rgba(255,255,255,0.08);
  }}
  .ch-card.collapsed .ch-card-body {{
    display: none;
  }}
  .ch-card.collapsed .ch-chevron {{
    transform: rotate(-90deg);
  }}
  .ch-chevron {{
    display: inline-block;
    font-size: 10px;
    color: var(--text-muted);
    transition: transform 0.15s ease;
  }}
  .slider-group {{
    margin-bottom: 7px;
  }}
  .slider-hdr {{
    display: flex;
    justify-content: space-between;
    font-size: 11.5px;
    color: var(--text-muted);
    margin-bottom: 2px;
  }}
  .slider-val {{
    color: var(--text-main);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }}
  input[type="range"] {{
    width: 100%;
    margin: 2px 0;
    accent-color: var(--accent);
    cursor: pointer;
  }}
  .two-col-sliders {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }}
  #hist-canvas {{
    width: 100%;
    height: 92px;
    background: #0b0e14;
    border: 1px solid var(--border-col);
    border-radius: 6px;
    display: block;
  }}
  .preset-info {{
    font-size: 11px;
    color: var(--text-muted);
    margin: 4px 0 8px 0;
  }}
</style>
</head>
<body>

<div class="studio-grid" id="studio-grid">
  <!-- LEFT: LARGE MAIN PICTURE VIEWPORT + INDIVIDUAL CHANNELS -->
  <div class="viewport-panel">
    <div class="viewport-toolbar">
      <div class="meta-badges">
        <span class="badge" style="background:#1e293b; border-color:#3b82f6; color:#f8fafc; font-weight:700; font-size:13px;">🔬 LifEasyLife</span>
        <span class="badge badge-highlight" id="res-badge">Original Size: {width} × {height} px (Full Quality)</span>
        <span class="badge">{meta.name}</span>
        <span class="badge">{mode_label}</span>
      </div>
      <div class="zoom-controls">
        <button class="btn" id="btn-undo-top" onclick="undoLastAction()" disabled title="Undo last change (Ctrl+Z or Cmd+Z)">↶ Undo</button>
        <button class="btn" id="btn-redo-top" onclick="redoLastAction()" disabled title="Redo change (Ctrl+Y or Cmd+Shift+Z)">↷ Redo</button>
        <span style="margin-left:4px;">View Size:</span>
        <button class="btn active" id="btn-fit" onclick="setZoomMode('fit')">Fit Large</button>
        <button class="btn" id="btn-fill" onclick="setZoomMode('fill')">Fill Width</button>
        <button class="btn" id="btn-1to1" onclick="setZoomMode('1to1')">100% Original Pixels (1:1)</button>
        <input type="range" id="zoom-slider" min="25" max="300" value="100" style="width:85px;" oninput="onCustomZoom(this.value)" />
        <span id="zoom-label" style="min-width:38px; text-align:right;">Fit</span>
        <button class="btn" id="btn-cinema" onclick="toggleCinemaMode()" title="Expand picture across full window width">⛶ Expand Screen</button>
      </div>
    </div>

    <div class="canvas-stage" id="canvas-stage">
      <canvas id="main-canvas" width="{width}" height="{height}"></canvas>
    </div>

    <!-- INDIVIDUAL CHANNELS STRIP -->
    <div class="strip-header">
      <div class="strip-title">🧪 Individual Channels (Preview & Direct Export)</div>
      <div class="btn-row">
        <button class="btn active" id="strip-mode-color" onclick="setStripMode('color')">Pseudocolor</button>
        <button class="btn" id="strip-mode-gray" onclick="setStripMode('gray')">Grayscale (High Contrast)</button>
      </div>
    </div>
    <div class="channel-strip" id="channel-strip"></div>
  </div>

  <!-- RIGHT: COLLAPSIBLE STUDIO DOCK -->
  <div class="dock-panel">

    <!-- PANEL 1: TWO DISTINCT EXPORT SECTIONS (ANALYSIS RAW vs PRESENTATION VISUAL) -->
    <div class="panel-box" id="panel-merge">
      <div class="panel-header" onclick="togglePanel('panel-merge')">
        <div class="panel-header-left">
          <span class="chevron">▼</span>
          <span>💾 1. Export: Analysis (Raw) & Presentation (Visual)</span>
        </div>
      </div>
      <div class="panel-body">
        <div class="form-row" style="margin-bottom:10px;">
          <label style="font-size:12px; color:var(--text-muted); white-space:nowrap;">File Name Prefix:</label>
          <input type="text" id="export-filename" class="text-input" value="{meta.name}" />
        </div>

        <!-- SECTION 1A: SAVE FOR ANALYSIS (RAW DATA) -->
        <div class="export-subbox export-subbox-analysis">
          <div class="subbox-title">
            <span style="color:#34d399;">🔬 1A. Save for Analysis (Raw Data)</span>
            <div class="btn-row">
              <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="setAllAnalysisChannels(true)">All</button>
              <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="setAllAnalysisChannels(false)">None</button>
            </div>
          </div>
          <div class="subbox-desc">
            Select channels to include/merge. Exports a <b>raw, unmodified scientific TIFF</b> preserving original <code>readlif</code> intensity values (8/16-bit). Completely bypasses LUTs, brightness, contrast, and pseudocolors. (Also available in the left sidebar via Python <code>tifffile</code> for 2D & full 3D Z-stacks.)
          </div>
          <div class="merge-list" id="analysis-checklist"></div>
          <button class="btn btn-success" style="width:100%;" onclick="downloadRawAnalysisTIFF()">
            ⬇️ Save for Analysis (Raw Unmodified TIFF)
          </button>
        </div>

        <!-- SECTION 1B: SAVE FOR PRESENTATION (VISUAL RENDER) -->
        <div class="export-subbox export-subbox-pres" style="margin-bottom:0;">
          <div class="subbox-title">
            <span style="color:#60a5fa;">🖼️ 1B. Save for Presentation (Visual Render)</span>
            <div class="btn-row">
              <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="setAllChannelsMerged(true)">All On</button>
              <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="setAllChannelsMerged(false)">All Off</button>
            </div>
          </div>
          <div class="subbox-desc">
            Saves the image <b>exactly as it currently looks on screen</b> with all Color Adjustment sliders, clipping, pseudocolors, and scale bar baked in (8-bit RGB, {width}×{height} px).
          </div>
          <div class="merge-list" id="merge-checklist"></div>

          <div class="form-row" style="margin-top:8px;">
            <span style="font-size:12px; font-weight:600;">Output Format:</span>
            <select id="pres-format-select" class="select-input" style="width:185px;">
              <option value="PNG">PNG (Lossless 8-bit RGB)</option>
              <option value="JPEG">JPEG (8-bit RGB)</option>
              <option value="TIFF">TIFF (8-bit RGB)</option>
            </select>
          </div>

          <div class="btn-row" style="margin-top:8px;">
            <button class="btn btn-primary" style="flex:1;" onclick="downloadPresentationImage()">
              ⬇️ Save for Presentation
            </button>
            <button class="btn" onclick="downloadAllSelectedChannelsPresentation()" title="Save each visible channel + merged composite in the selected presentation format">
              ⬇️ Save Each Ch + Merge
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- PANEL 2: COLOR ADJUSTMENT -->
    <div class="panel-box" id="panel-color">
      <div class="panel-header" onclick="togglePanel('panel-color')">
        <div class="panel-header-left">
          <span class="chevron">▼</span>
          <span>🎨 2. Color Adjustment</span>
        </div>
        <div class="btn-row" onclick="event.stopPropagation();">
          <button class="btn" id="btn-undo-panel" onclick="undoLastAction()" disabled title="Undo (Ctrl+Z or Cmd+Z)">↶ Undo</button>
          <button class="btn" onclick="autoContrastAll()">✨ Auto All</button>
          <button class="btn" onclick="resetAllChannels()">↺ Reset All</button>
        </div>
      </div>
      <div class="panel-body">
        <canvas id="hist-canvas" width="380" height="92"></canvas>
        <div style="font-size:11px; color:var(--text-muted); margin-top:4px; margin-bottom:8px;">
          Drag any slider below — updates live in real time. Press <b>Ctrl+Z</b> (or <b>Cmd+Z</b>) to undo.
        </div>
        <div id="channels-container"></div>
      </div>
    </div>

    <!-- PANEL 3: SCALE BAR & OBJECTIVE LENS PANEL -->
    <div class="panel-box" id="panel-scalebar">
      <div class="panel-header" onclick="togglePanel('panel-scalebar')">
        <div class="panel-header-left">
          <span class="chevron">▼</span>
          <span>📏 3. Scale Bar & Objective Lens</span>
        </div>
        <label style="font-size:12px; font-weight:normal; cursor:pointer; display:flex; align-items:center; gap:5px;" onclick="event.stopPropagation();">
          <input type="checkbox" id="sb-enabled" checked onchange="onScaleBarChange()" /> Show
        </label>
      </div>
      <div class="panel-body">
        <div class="cal-info-box" id="cal-info-box"></div>

        <div class="form-row">
          <span>Objective Lens:</span>
          <div class="btn-row">
            <button class="btn" id="obj-10x" onclick="selectObjective('10x')">10x</button>
            <button class="btn" id="obj-40x" onclick="selectObjective('40x')">40x</button>
            <button class="btn" id="obj-60x" onclick="selectObjective('60x')">60x</button>
          </div>
        </div>

        <div class="form-row">
          <span>Scale Bar Length (µm):</span>
          <div style="display:flex; align-items:center; gap:5px;">
            <input type="number" id="sb-um" class="num-input" min="0.1" max="5000" step="any" value="25" oninput="onScaleBarChange()" />
            <span style="font-size:12px; font-weight:600;">µm</span>
          </div>
        </div>

        <div class="btn-row" style="margin-bottom:8px; justify-content:flex-end;">
          <span style="font-size:11px; color:var(--text-muted); align-self:center; margin-right:auto;">Quick µm:</span>
          <button class="btn" onclick="setQuickUm(10)">10 µm</button>
          <button class="btn" onclick="setQuickUm(20)">20 µm</button>
          <button class="btn" onclick="setQuickUm(25)">25 µm</button>
          <button class="btn" onclick="setQuickUm(50)">50 µm</button>
          <button class="btn" onclick="setQuickUm(100)">100 µm</button>
        </div>

        <div class="form-row">
          <span title="Calibration ratio in pixels per micrometer for the selected objective">Calibration (px / µm):</span>
          <div style="display:flex; gap:5px; align-items:center;">
            <input type="number" id="sb-px-per-um" class="num-input" min="0.001" step="0.001" value="1.0" oninput="onCustomCalibrationInput()" />
            <button class="btn" onclick="resetCalibrationToLifDefault()" title="Restore exact Leica .lif calibration for this objective">Reset Cal</button>
          </div>
        </div>

        <div class="form-row" style="margin-bottom:0;">
          <span>Corner & Text Size:</span>
          <div style="display:flex; gap:6px; align-items:center;">
            <select id="sb-pos" class="select-input" style="width:130px;" onchange="onScaleBarChange()">
              <option value="br">Bottom Right</option>
              <option value="bl">Bottom Left</option>
              <option value="tr">Top Right</option>
              <option value="tl">Top Left</option>
            </select>
            <input type="range" id="sb-thickness" min="50" max="250" value="100" style="width:75px;" title="Scale bar & text size" oninput="onScaleBarChange()" />
          </div>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
const CONFIG = {config_json};

// ============================================================================
// STRICT ISOLATION OF ARRAYS:
// 1. `visPlanesU16`: used ONLY for 8-bit RGB Presentation rendering + LUTs
// 2. `rawAnalysisBuffers`: frozen, unmodified raw integer bytes from `readlif`
//    used ONLY for 'Save for Analysis (Raw Data)'
// ============================================================================
const visPlanesU16 = [];
const rawAnalysisBuffers = [];

for (let c = 0; c < CONFIG.numChannels; c++) {{
  // Presentation LUT index plane
  const b64Vis = CONFIG.channels[c].b64U16;
  const binVis = atob(b64Vis);
  const bytesVis = new Uint8Array(binVis.length);
  for (let i = 0; i < binVis.length; i++) {{
    bytesVis[i] = binVis.charCodeAt(i);
  }}
  visPlanesU16.push(new Uint16Array(bytesVis.buffer));

  // Strictly isolated Raw Analysis buffer (exact unmodified readlif bytes)
  const b64Raw = CONFIG.channels[c].b64RawExact;
  const binRaw = atob(b64Raw);
  const bytesRaw = new Uint8Array(binRaw.length);
  for (let i = 0; i < binRaw.length; i++) {{
    bytesRaw[i] = binRaw.charCodeAt(i);
  }}
  // Freeze raw analysis metadata wrapper so it cannot be mutated
  rawAnalysisBuffers.push(Object.freeze({{
    bits: CONFIG.channels[c].rawContainerBits,
    byteView: bytesRaw,
    typedView: CONFIG.channels[c].rawContainerBits === 8
      ? bytesRaw
      : new Uint16Array(bytesRaw.buffer)
  }}));
}}

// Per-channel 65536-entry Lookup Tables for Presentation rendering ONLY
const lutNorm = [];
for (let c = 0; c < CONFIG.numChannels; c++) {{
  lutNorm.push(new Float32Array(65536));
}}

const defaultObj = CONFIG.defaultObjective || '40x';
const defaultPxPerUm = CONFIG.objectiveDefaults[defaultObj] || 1.0;

let state = {{
  zoomMode: 'fit',
  customZoom: 100,
  cinemaMode: false,
  stripMode: 'color',
  analysisSelected: new Array(CONFIG.numChannels).fill(true),
  scaleBar: {{
    enabled: true,
    objective: defaultObj,
    um: 25,
    pxPerUm: defaultPxPerUm,
    customObjCal: {{ ...CONFIG.objectiveDefaults }},
    pos: 'br',
    sizeScale: 100
  }},
  channels: []
}};

function createDefaultChannelState(c) {{
  const chMeta = CONFIG.channels[c];
  return {{
    enabled: true,
    preset: chMeta.defaultPreset,
    hex: chMeta.defaultHex,
    brightness: 0,
    contrast: 0,
    exposure: 0,
    gamma: 1.0,
    blackPoint: chMeta.rawMin,
    whitePoint: chMeta.rawMax,
    invert: false
  }};
}}

function loadSavedState() {{
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    state.channels.push(createDefaultChannelState(c));
  }}
  try {{
    const raw = localStorage.getItem(CONFIG.storageKey);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (Array.isArray(saved.analysisSelected) && saved.analysisSelected.length === CONFIG.numChannels) {{
      state.analysisSelected = saved.analysisSelected.map(Boolean);
    }}
    if (saved.scaleBar) {{
      const savedObj = saved.scaleBar.objective || defaultObj;
      const mergedCal = {{ ...CONFIG.objectiveDefaults, ...(saved.scaleBar.customObjCal || {{}}) }};
      state.scaleBar = {{
        ...state.scaleBar,
        ...saved.scaleBar,
        objective: savedObj,
        customObjCal: mergedCal,
        pxPerUm: mergedCal[savedObj] || defaultPxPerUm
      }};
    }}
    if (Array.isArray(saved.channels) && saved.channels.length === CONFIG.numChannels) {{
      for (let c = 0; c < CONFIG.numChannels; c++) {{
        const chMeta = CONFIG.channels[c];
        const sc = saved.channels[c];
        state.channels[c] = {{
          ...state.channels[c],
          ...sc,
          blackPoint: Math.max(0, Math.min(chMeta.sliderMax, Number(sc.blackPoint ?? chMeta.rawMin))),
          whitePoint: Math.max(0.1, Math.min(chMeta.sliderMax, Number(sc.whitePoint ?? chMeta.rawMax)))
        }};
      }}
    }}
  }} catch (e) {{
    console.warn("Could not load saved state:", e);
  }}
}}

function saveState() {{
  try {{
    localStorage.setItem(CONFIG.storageKey, JSON.stringify({{
      analysisSelected: state.analysisSelected,
      scaleBar: state.scaleBar,
      channels: state.channels
    }}));
  }} catch (e) {{}}
}}

// ============================================================================
// UNDO (Ctrl+Z / Cmd+Z) & REDO (Ctrl+Y / Cmd+Shift+Z) HISTORY STACK
// ============================================================================
const undoStack = [];
const redoStack = [];
let activeDragSnapshot = null;

function getStateSnapshot() {{
  return JSON.stringify({{
    analysisSelected: state.analysisSelected,
    scaleBar: state.scaleBar,
    channels: state.channels
  }});
}}

function beginContinuousEdit() {{
  if (activeDragSnapshot === null) {{
    activeDragSnapshot = getStateSnapshot();
  }}
}}

function commitContinuousEdit() {{
  if (activeDragSnapshot !== null) {{
    pushUndoSnapshot(activeDragSnapshot);
    activeDragSnapshot = null;
  }}
}}

function recordDiscreteEdit() {{
  commitContinuousEdit();
  pushUndoSnapshot(getStateSnapshot());
}}

function pushUndoSnapshot(snapStr) {{
  if (!snapStr || snapStr === getStateSnapshot()) return;
  if (undoStack.length > 0 && undoStack[undoStack.length - 1] === snapStr) return;
  undoStack.push(snapStr);
  if (undoStack.length > 60) undoStack.shift();
  redoStack.length = 0;
  updateUndoRedoButtons();
}}

function updateUndoRedoButtons() {{
  const canUndo = undoStack.length > 0 || activeDragSnapshot !== null;
  const canRedo = redoStack.length > 0;
  const uTop = document.getElementById("btn-undo-top");
  const rTop = document.getElementById("btn-redo-top");
  const uPanel = document.getElementById("btn-undo-panel");
  if (uTop) uTop.disabled = !canUndo;
  if (rTop) rTop.disabled = !canRedo;
  if (uPanel) uPanel.disabled = !canUndo;
}}

function applyStateSnapshot(snapStr) {{
  try {{
    const snap = JSON.parse(snapStr);
    if (Array.isArray(snap.analysisSelected)) {{
      state.analysisSelected = snap.analysisSelected.map(Boolean);
    }}
    if (snap.scaleBar) {{
      state.scaleBar = {{ ...state.scaleBar, ...snap.scaleBar }};
    }}
    if (Array.isArray(snap.channels)) {{
      state.channels = snap.channels.map(ch => ({{ ...ch }}));
    }}
    syncScaleBarUI();
    for (let c = 0; c < CONFIG.numChannels; c++) {{
      const aChk = document.getElementById(`analysis-chk-${{c}}`);
      if (aChk) aChk.checked = state.analysisSelected[c];
      const tCard = document.getElementById(`thumb-card-${{c}}`);
      if (tCard) tCard.className = "ch-thumb-card" + (state.channels[c].enabled ? "" : " muted");
      syncChannelSlidersDOM(c);
      updateChannelLUT(c);
    }}
    saveState();
    requestRender();
  }} catch (e) {{
    console.warn("Failed to apply undo/redo snapshot:", e);
  }}
}}

function undoLastAction() {{
  if (activeDragSnapshot !== null) {{
    pushUndoSnapshot(activeDragSnapshot);
    activeDragSnapshot = null;
  }}
  if (undoStack.length === 0) return;
  const currentSnap = getStateSnapshot();
  const prevSnap = undoStack.pop();
  redoStack.push(currentSnap);
  applyStateSnapshot(prevSnap);
  updateUndoRedoButtons();
}}

function redoLastAction() {{
  commitContinuousEdit();
  if (redoStack.length === 0) return;
  const currentSnap = getStateSnapshot();
  const nextSnap = redoStack.pop();
  undoStack.push(currentSnap);
  applyStateSnapshot(nextSnap);
  updateUndoRedoButtons();
}}

function handleGlobalKeyDown(e) {{
  const activeTag = document.activeElement ? document.activeElement.id : "";
  if (activeTag === "export-filename") return;

  const isCtrlOrCmd = e.ctrlKey || e.metaKey;
  if (!isCtrlOrCmd) return;

  const key = (e.key || "").toLowerCase();
  if (key === "z" && !e.shiftKey) {{
    e.preventDefault();
    undoLastAction();
  }} else if ((key === "z" && e.shiftKey) || key === "y") {{
    e.preventDefault();
    redoLastAction();
  }}
}}

function togglePanel(panelId) {{
  const el = document.getElementById(panelId);
  if (el) el.classList.toggle("collapsed");
}}

function toggleChannelCard(c) {{
  const el = document.getElementById(`ch-card-${{c}}`);
  if (el) el.classList.toggle("collapsed");
}}

function hexToRgbNorm(hex) {{
  let clean = (hex || "#ffffff").replace("#", "").trim();
  if (clean.length === 3) clean = clean[0]+clean[0]+clean[1]+clean[1]+clean[2]+clean[2];
  const num = parseInt(clean, 16);
  if (isNaN(num)) return [1, 1, 1];
  return [((num >> 16) & 255) / 255, ((num >> 8) & 255) / 255, (num & 255) / 255];
}}

function rawToU16(c, rawVal) {{
  const m = CONFIG.channels[c];
  return Math.max(0, Math.min(65535, Math.round((rawVal - m.offsetToU16) * m.scaleToU16)));
}}

function updateChannelLUT(c) {{
  const cfg = state.channels[c];
  const m = CONFIG.channels[c];
  const lut = lutNorm[c];
  const maxU16 = m.u16SliderMax;

  const bpU16 = rawToU16(c, cfg.blackPoint);
  const wpU16 = rawToU16(c, cfg.whitePoint);
  const denom = Math.max(1e-6, wpU16 - bpU16);

  const expFactor = Math.abs(cfg.exposure) > 1e-4 ? Math.pow(2.0, cfg.exposure) : 1.0;
  const hasBC = Math.abs(cfg.contrast) > 1e-4 || Math.abs(cfg.brightness) > 1e-4;
  const cNorm = Math.max(-0.98, Math.min(0.98, cfg.contrast / 100.0));
  const contrastFactor = Math.tan((cNorm + 1.0) * (Math.PI / 4.0));
  const bOffset = (cfg.brightness / 100.0) * 0.5;
  const invGamma = 1.0 / Math.max(0.05, cfg.gamma);
  const hasGamma = Math.abs(cfg.gamma - 1.0) > 1e-4;
  const invert = cfg.invert;

  for (let v = 0; v <= maxU16; v++) {{
    let norm = (v - bpU16) / denom;
    if (norm < 0.0) norm = 0.0;
    else if (norm > 1.0) norm = 1.0;

    if (expFactor !== 1.0) {{
      norm *= expFactor;
    }}
    if (hasBC) {{
      norm = contrastFactor * (norm - 0.5 + bOffset) + 0.5;
    }}
    if (norm < 0.0) norm = 0.0;
    else if (norm > 1.0) norm = 1.0;

    if (hasGamma) {{
      norm = Math.pow(norm, invGamma);
    }}
    if (invert) {{
      norm = 1.0 - norm;
    }}
    lut[v] = norm;
  }}
}}

function buildUI() {{
  syncScaleBarUI();
  buildAnalysisChecklist();
  buildMergeChecklist();
  buildChannelStripCards();
  buildChannelControlCards();
}}

function buildAnalysisChecklist() {{
  const container = document.getElementById("analysis-checklist");
  container.innerHTML = "";
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const chMeta = CONFIG.channels[c];
    const isSel = state.analysisSelected[c];
    const row = document.createElement("div");
    row.className = "merge-item";
    row.innerHTML = `
      <label class="merge-item-left">
        <input type="checkbox" id="analysis-chk-${{c}}" ${{isSel ? 'checked' : ''}} onchange="toggleAnalysisChannel(${{c}}, this.checked)" />
        <span style="font-weight:600;">${{chMeta.name}}</span>
      </label>
      <span style="font-size:11px; color:#34d399; font-variant-numeric:tabular-nums;">
        Raw ${{chMeta.bitDepth}}-bit [${{chMeta.rawMin}}–${{chMeta.rawMax}}]
      </span>
    `;
    container.appendChild(row);
  }}
}}

function toggleAnalysisChannel(c, checked) {{
  recordDiscreteEdit();
  state.analysisSelected[c] = checked;
  const el = document.getElementById(`analysis-chk-${{c}}`);
  if (el) el.checked = checked;
  saveState();
}}

function setAllAnalysisChannels(checked) {{
  recordDiscreteEdit();
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    state.analysisSelected[c] = checked;
    const el = document.getElementById(`analysis-chk-${{c}}`);
    if (el) el.checked = checked;
  }}
  saveState();
}}

function buildMergeChecklist() {{
  const container = document.getElementById("merge-checklist");
  container.innerHTML = "";
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const cfg = state.channels[c];
    const row = document.createElement("div");
    row.className = "merge-item";
    row.innerHTML = `
      <label class="merge-item-left">
        <input type="checkbox" id="merge-chk-${{c}}" ${{cfg.enabled ? 'checked' : ''}} onchange="toggleChannelEnabled(${{c}}, this.checked)" />
        <span class="color-dot" id="merge-dot-${{c}}" style="background:${{cfg.hex}};"></span>
        <span style="font-weight:600;">Ch ${{c + 1}}:</span>
        <span id="merge-preset-lbl-${{c}}" style="color:var(--text-muted);">${{cfg.preset}}</span>
      </label>
      <button class="btn" style="padding:2px 7px; font-size:11px;" onclick="soloChannel(${{c}})" title="Show only this channel">Solo</button>
    `;
    container.appendChild(row);
  }}
}}

function buildChannelStripCards() {{
  const strip = document.getElementById("channel-strip");
  strip.innerHTML = "";
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const cfg = state.channels[c];
    const card = document.createElement("div");
    card.className = "ch-thumb-card" + (cfg.enabled ? "" : " muted");
    card.id = `thumb-card-${{c}}`;
    card.innerHTML = `
      <div class="ch-thumb-hdr">
        <span><span class="color-dot" id="thumb-dot-${{c}}" style="background:${{cfg.hex}}; margin-right:5px;"></span>Ch ${{c + 1}}: <span id="thumb-title-${{c}}">${{cfg.preset}}</span></span>
        <div class="btn-row">
          <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="downloadSingleChannelRawTIFF(${{c}})" title="Save Raw Unmodified Analysis TIFF of Channel ${{c + 1}}">🔬 Raw</button>
          <button class="btn" style="padding:2px 6px; font-size:11px;" onclick="downloadSingleChannelPresentation(${{c}})" title="Save Visual Presentation Image of Channel ${{c + 1}}">🖼️ Visual</button>
        </div>
      </div>
      <canvas id="thumb-canvas-${{c}}" class="ch-thumb-canvas" width="{width}" height="{height}"></canvas>
    `;
    strip.appendChild(card);
  }}
}}

function buildChannelControlCards() {{
  const container = document.getElementById("channels-container");
  container.innerHTML = "";
  const presetNames = Object.keys(CONFIG.presets);

  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const chMeta = CONFIG.channels[c];
    const cfg = state.channels[c];
    const stepVal = Math.max(0.1, Number(((chMeta.sliderMax) / 500).toFixed(2)));

    const optionsHtml = presetNames.map(p =>
      `<option value="${{p}}" ${{p === cfg.preset ? 'selected' : ''}}>${{p}}</option>`
    ).join("");

    const card = document.createElement("div");
    card.className = "ch-card";
    card.id = `ch-card-${{c}}`;
    card.innerHTML = `
      <div class="ch-card-hdr" onclick="toggleChannelCard(${{c}})">
        <div class="ch-card-title">
          <span class="ch-chevron">▼</span>
          <input type="checkbox" id="card-chk-${{c}}" ${{cfg.enabled ? 'checked' : ''}} onclick="event.stopPropagation();" onchange="toggleChannelEnabled(${{c}}, this.checked)" title="Include channel in merged presentation picture" />
          <span class="color-dot" id="card-dot-${{c}}" style="background:${{cfg.hex}};"></span>
          <span>${{chMeta.name}}</span>
        </div>
        <div class="btn-row" onclick="event.stopPropagation();">
          <button class="btn" style="padding:3px 7px; font-size:11px;" onclick="autoContrastChannel(${{c}})">Auto</button>
          <button class="btn" style="padding:3px 7px; font-size:11px;" onclick="resetChannel(${{c}})">Reset</button>
        </div>
      </div>

      <div class="ch-card-body">
        <div style="display:grid; grid-template-columns: 1fr 42px; gap:6px; align-items:center; margin-bottom:4px;">
          <select id="preset-sel-${{c}}" class="select-input" onchange="onPresetSelect(${{c}}, this.value)">
            ${{optionsHtml}}
          </select>
          <input type="color" id="color-pick-${{c}}" value="${{cfg.hex}}" style="width:42px; height:29px; padding:0; border:1px solid var(--border-col); border-radius:6px; background:transparent; cursor:pointer;" oninput="onColorPick(${{c}}, this.value)" />
        </div>
        <div class="preset-info" id="preset-info-${{c}}"></div>

        <div class="two-col-sliders">
          <div class="slider-group">
            <div class="slider-hdr"><span>Brightness</span><span class="slider-val" id="val-bright-${{c}}">${{cfg.brightness}}</span></div>
            <input type="range" id="sl-bright-${{c}}" min="-100" max="100" step="1" value="${{cfg.brightness}}" oninput="onSliderInput(${{c}}, 'brightness', this.value)" />
          </div>
          <div class="slider-group">
            <div class="slider-hdr"><span>Contrast</span><span class="slider-val" id="val-cont-${{c}}">${{cfg.contrast}}</span></div>
            <input type="range" id="sl-cont-${{c}}" min="-100" max="100" step="1" value="${{cfg.contrast}}" oninput="onSliderInput(${{c}}, 'contrast', this.value)" />
          </div>
          <div class="slider-group">
            <div class="slider-hdr"><span>Exposure (EV)</span><span class="slider-val" id="val-exp-${{c}}">${{Number(cfg.exposure).toFixed(2)}}</span></div>
            <input type="range" id="sl-exp-${{c}}" min="-3.0" max="3.0" step="0.05" value="${{cfg.exposure}}" oninput="onSliderInput(${{c}}, 'exposure', this.value)" />
          </div>
          <div class="slider-group">
            <div class="slider-hdr"><span>Gamma (γ)</span><span class="slider-val" id="val-gam-${{c}}">${{Number(cfg.gamma).toFixed(2)}}</span></div>
            <input type="range" id="sl-gam-${{c}}" min="0.10" max="3.00" step="0.05" value="${{cfg.gamma}}" oninput="onSliderInput(${{c}}, 'gamma', this.value)" />
          </div>
          <div class="slider-group">
            <div class="slider-hdr"><span>Black Point</span><span class="slider-val" id="val-bp-${{c}}">${{Math.round(cfg.blackPoint)}}</span></div>
            <input type="range" id="sl-bp-${{c}}" min="0" max="${{chMeta.sliderMax}}" step="${{stepVal}}" value="${{cfg.blackPoint}}" oninput="onSliderInput(${{c}}, 'blackPoint', this.value)" />
          </div>
          <div class="slider-group">
            <div class="slider-hdr"><span>White Point</span><span class="slider-val" id="val-wp-${{c}}">${{Math.round(cfg.whitePoint)}}</span></div>
            <input type="range" id="sl-wp-${{c}}" min="0" max="${{chMeta.sliderMax}}" step="${{stepVal}}" value="${{cfg.whitePoint}}" oninput="onSliderInput(${{c}}, 'whitePoint', this.value)" />
          </div>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:11.5px; color:var(--text-muted); margin-top:2px;">
          <span>Raw Range: [${{chMeta.rawMin}} – ${{chMeta.rawMax}}] (${{chMeta.bitDepth}}-bit)</span>
          <label style="cursor:pointer;"><input type="checkbox" id="chk-inv-${{c}}" ${{cfg.invert ? 'checked' : ''}} onchange="onInvertToggle(${{c}}, this.checked)" /> Invert LUT</label>
        </div>
      </div>
    `;
    container.appendChild(card);
    updatePresetDescription(c);
  }}
}}

function updatePresetDescription(c) {{
  const cfg = state.channels[c];
  const p = CONFIG.presets[cfg.preset];
  const el = document.getElementById(`preset-info-${{c}}`);
  if (!el) return;
  if (p && (p.ex || p.em)) {{
    el.innerHTML = `<span style="color:${{cfg.hex}}; font-weight:600;">Ex ${{p.ex}} nm / Em ${{p.em}} nm</span> — ${{p.desc}}`;
  }} else if (p) {{
    el.textContent = p.desc;
  }} else {{
    el.textContent = "";
  }}
}}

let renderPending = false;
function requestRender() {{
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(() => {{
    renderPending = false;
    renderAll();
  }});
}}

function onSliderInput(c, field, val) {{
  beginContinuousEdit();
  const num = parseFloat(val);
  const cfg = state.channels[c];
  if (field === 'blackPoint') {{
    cfg.blackPoint = Math.min(num, cfg.whitePoint - 0.5);
    document.getElementById(`val-bp-${{c}}`).textContent = Math.round(cfg.blackPoint);
  }} else if (field === 'whitePoint') {{
    cfg.whitePoint = Math.max(num, cfg.blackPoint + 0.5);
    document.getElementById(`val-wp-${{c}}`).textContent = Math.round(cfg.whitePoint);
  }} else if (field === 'brightness') {{
    cfg.brightness = Math.round(num);
    document.getElementById(`val-bright-${{c}}`).textContent = cfg.brightness;
  }} else if (field === 'contrast') {{
    cfg.contrast = Math.round(num);
    document.getElementById(`val-cont-${{c}}`).textContent = cfg.contrast;
  }} else if (field === 'exposure') {{
    cfg.exposure = num;
    document.getElementById(`val-exp-${{c}}`).textContent = num.toFixed(2);
  }} else if (field === 'gamma') {{
    cfg.gamma = num;
    document.getElementById(`val-gam-${{c}}`).textContent = num.toFixed(2);
  }}
  updateChannelLUT(c);
  updateUndoRedoButtons();
  saveState();
  requestRender();
}}

function onInvertToggle(c, checked) {{
  recordDiscreteEdit();
  state.channels[c].invert = checked;
  updateChannelLUT(c);
  saveState();
  requestRender();
}}

function onPresetSelect(c, presetName) {{
  recordDiscreteEdit();
  const cfg = state.channels[c];
  cfg.preset = presetName;
  if (CONFIG.presets[presetName] && presetName !== "Custom Color") {{
    cfg.hex = CONFIG.presets[presetName].hex;
    document.getElementById(`color-pick-${{c}}`).value = cfg.hex;
  }}
  syncChannelColorUI(c);
  updatePresetDescription(c);
  saveState();
  requestRender();
}}

function onColorPick(c, hexVal) {{
  beginContinuousEdit();
  const cfg = state.channels[c];
  cfg.hex = hexVal;
  syncChannelColorUI(c);
  updateUndoRedoButtons();
  saveState();
  requestRender();
}}

function syncChannelColorUI(c) {{
  const cfg = state.channels[c];
  const ids = [`merge-dot-${{c}}`, `thumb-dot-${{c}}`, `card-dot-${{c}}`];
  ids.forEach(id => {{
    const el = document.getElementById(id);
    if (el) el.style.background = cfg.hex;
  }});
  const mLbl = document.getElementById(`merge-preset-lbl-${{c}}`);
  if (mLbl) mLbl.textContent = cfg.preset;
  const tLbl = document.getElementById(`thumb-title-${{c}}`);
  if (tLbl) tLbl.textContent = cfg.preset;
}}

function _setChannelEnabledSilent(c, enabled) {{
  state.channels[c].enabled = enabled;
  const mChk = document.getElementById(`merge-chk-${{c}}`);
  if (mChk) mChk.checked = enabled;
  const cChk = document.getElementById(`card-chk-${{c}}`);
  if (cChk) cChk.checked = enabled;
  const tCard = document.getElementById(`thumb-card-${{c}}`);
  if (tCard) tCard.className = "ch-thumb-card" + (enabled ? "" : " muted");
}}

function toggleChannelEnabled(c, enabled) {{
  recordDiscreteEdit();
  _setChannelEnabledSilent(c, enabled);
  saveState();
  requestRender();
}}

function setAllChannelsMerged(enabled) {{
  recordDiscreteEdit();
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    _setChannelEnabledSilent(c, enabled);
  }}
  saveState();
  requestRender();
}}

function soloChannel(targetIdx) {{
  recordDiscreteEdit();
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    _setChannelEnabledSilent(c, c === targetIdx);
  }}
  saveState();
  requestRender();
}}

function syncChannelSlidersDOM(c) {{
  const cfg = state.channels[c];
  document.getElementById(`card-chk-${{c}}`).checked = cfg.enabled;
  document.getElementById(`merge-chk-${{c}}`).checked = cfg.enabled;
  document.getElementById(`preset-sel-${{c}}`).value = cfg.preset;
  document.getElementById(`color-pick-${{c}}`).value = cfg.hex;
  document.getElementById(`sl-bright-${{c}}`).value = cfg.brightness;
  document.getElementById(`val-bright-${{c}}`).textContent = cfg.brightness;
  document.getElementById(`sl-cont-${{c}}`).value = cfg.contrast;
  document.getElementById(`val-cont-${{c}}`).textContent = cfg.contrast;
  document.getElementById(`sl-exp-${{c}}`).value = cfg.exposure;
  document.getElementById(`val-exp-${{c}}`).textContent = Number(cfg.exposure).toFixed(2);
  document.getElementById(`sl-gam-${{c}}`).value = cfg.gamma;
  document.getElementById(`val-gam-${{c}}`).textContent = Number(cfg.gamma).toFixed(2);
  document.getElementById(`sl-bp-${{c}}`).value = cfg.blackPoint;
  document.getElementById(`val-bp-${{c}}`).textContent = Math.round(cfg.blackPoint);
  document.getElementById(`sl-wp-${{c}}`).value = cfg.whitePoint;
  document.getElementById(`val-wp-${{c}}`).textContent = Math.round(cfg.whitePoint);
  document.getElementById(`chk-inv-${{c}}`).checked = cfg.invert;
  syncChannelColorUI(c);
  updatePresetDescription(c);
}}

function resetChannel(c) {{
  recordDiscreteEdit();
  state.channels[c] = createDefaultChannelState(c);
  syncChannelSlidersDOM(c);
  updateChannelLUT(c);
  saveState();
  requestRender();
}}

function resetAllChannels() {{
  recordDiscreteEdit();
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    state.channels[c] = createDefaultChannelState(c);
    syncChannelSlidersDOM(c);
    updateChannelLUT(c);
  }}
  saveState();
  requestRender();
}}

function autoContrastChannel(c) {{
  recordDiscreteEdit();
  const chMeta = CONFIG.channels[c];
  const cfg = state.channels[c];
  cfg.blackPoint = chMeta.autoLow;
  cfg.whitePoint = chMeta.autoHigh;
  cfg.brightness = 0;
  cfg.contrast = 0;
  cfg.exposure = 0;
  cfg.gamma = 1.0;
  syncChannelSlidersDOM(c);
  updateChannelLUT(c);
  saveState();
  requestRender();
}}

function autoContrastAll() {{
  recordDiscreteEdit();
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const chMeta = CONFIG.channels[c];
    const cfg = state.channels[c];
    cfg.blackPoint = chMeta.autoLow;
    cfg.whitePoint = chMeta.autoHigh;
    syncChannelSlidersDOM(c);
    updateChannelLUT(c);
  }}
  saveState();
  requestRender();
}}

// Objective lens & Scale Bar handlers
function updateCalibrationInfoBox() {{
  const sb = state.scaleBar;
  const box = document.getElementById("cal-info-box");
  if (!box) return;
  const pxPerUm = sb.pxPerUm > 0 ? sb.pxPerUm : 1.0;
  const umPerPx = 1.0 / pxPerUm;
  const fovW = (CONFIG.width - 1) / pxPerUm;
  const fovH = (CONFIG.height - 1) / pxPerUm;
  const barPx = Math.round(sb.um * pxPerUm);

  box.innerHTML =
    `<div><b>.lif Metadata:</b> ${{CONFIG.detectedSummary}}</div>` +
    `<div><b>Active (${{sb.objective}}):</b> ${{pxPerUm.toFixed(4)}} px/µm (${{umPerPx.toFixed(4)}} µm/px) • FOV: ${{fovW.toFixed(1)}} × ${{fovH.toFixed(1)}} µm</div>` +
    `<div><b>Scale Bar:</b> ${{sb.um}} µm = <b>${{barPx}} px</b> (on ${{CONFIG.width}}×${{CONFIG.height}} px image)</div>`;
}}

function syncScaleBarUI() {{
  const sb = state.scaleBar;
  document.getElementById("sb-enabled").checked = sb.enabled;
  document.getElementById("sb-um").value = sb.um;
  document.getElementById("sb-px-per-um").value = Number(sb.pxPerUm).toFixed(4);
  document.getElementById("sb-pos").value = sb.pos;
  document.getElementById("sb-thickness").value = sb.sizeScale;

  ['10x', '40x', '60x'].forEach(k => {{
    const btn = document.getElementById(`obj-${{k}}`);
    if (btn) btn.className = "btn" + (sb.objective === k ? " active" : "");
  }});
  updateCalibrationInfoBox();
}}

function selectObjective(objKey) {{
  recordDiscreteEdit();
  const sb = state.scaleBar;
  sb.objective = objKey;
  sb.pxPerUm = sb.customObjCal[objKey] || CONFIG.objectiveDefaults[objKey] || 1.0;
  syncScaleBarUI();
  saveState();
  requestRender();
}}

function resetCalibrationToLifDefault() {{
  recordDiscreteEdit();
  const sb = state.scaleBar;
  sb.customObjCal = {{ ...CONFIG.objectiveDefaults }};
  sb.pxPerUm = CONFIG.objectiveDefaults[sb.objective] || 1.0;
  syncScaleBarUI();
  saveState();
  requestRender();
}}

function setQuickUm(umVal) {{
  recordDiscreteEdit();
  state.scaleBar.um = umVal;
  document.getElementById("sb-um").value = umVal;
  updateCalibrationInfoBox();
  saveState();
  requestRender();
}}

function onCustomCalibrationInput() {{
  beginContinuousEdit();
  const val = parseFloat(document.getElementById("sb-px-per-um").value);
  if (!isNaN(val) && val > 0) {{
    state.scaleBar.pxPerUm = val;
    state.scaleBar.customObjCal[state.scaleBar.objective] = val;
    updateCalibrationInfoBox();
    updateUndoRedoButtons();
    saveState();
    requestRender();
  }}
}}

function onScaleBarChange() {{
  beginContinuousEdit();
  const sb = state.scaleBar;
  sb.enabled = document.getElementById("sb-enabled").checked;
  const umVal = parseFloat(document.getElementById("sb-um").value);
  if (!isNaN(umVal) && umVal > 0) sb.um = umVal;
  sb.pos = document.getElementById("sb-pos").value;
  sb.sizeScale = parseFloat(document.getElementById("sb-thickness").value) || 100;
  updateCalibrationInfoBox();
  updateUndoRedoButtons();
  saveState();
  requestRender();
}}

// Viewport zoom & display modes
function setZoomMode(mode) {{
  state.zoomMode = mode;
  ['fit', 'fill', '1to1'].forEach(m => {{
    const btn = document.getElementById(`btn-${{m}}`);
    if (btn) btn.className = "btn" + (mode === m ? " active" : "");
  }});
  applyCanvasDisplaySize();
}}

function onCustomZoom(val) {{
  state.zoomMode = 'custom';
  state.customZoom = parseInt(val, 10) || 100;
  ['fit', 'fill', '1to1'].forEach(m => {{
    const btn = document.getElementById(`btn-${{m}}`);
    if (btn) btn.className = "btn";
  }});
  applyCanvasDisplaySize();
}}

function toggleCinemaMode() {{
  state.cinemaMode = !state.cinemaMode;
  const grid = document.getElementById("studio-grid");
  const btn = document.getElementById("btn-cinema");
  if (state.cinemaMode) {{
    grid.classList.add("cinema-mode");
    btn.classList.add("active");
    btn.textContent = "⛶ Exit Expand";
  }} else {{
    grid.classList.remove("cinema-mode");
    btn.classList.remove("active");
    btn.textContent = "⛶ Expand Screen";
  }}
  applyCanvasDisplaySize();
}}

function applyCanvasDisplaySize() {{
  const canvas = document.getElementById("main-canvas");
  const stage = document.getElementById("canvas-stage");
  const zoomLbl = document.getElementById("zoom-label");

  canvas.classList.remove("pixelated");
  if (state.zoomMode === 'fit') {{
    const availW = Math.max(320, stage.clientWidth - 24);
    const availH = Math.max(420, window.innerHeight * 0.78);
    const scale = Math.min(availW / CONFIG.width, availH / CONFIG.height);
    canvas.style.width = `${{Math.round(CONFIG.width * scale)}}px`;
    canvas.style.height = `${{Math.round(CONFIG.height * scale)}}px`;
    zoomLbl.textContent = `${{Math.round(scale * 100)}}%`;
    document.getElementById("zoom-slider").value = Math.min(300, Math.max(25, Math.round(scale * 100)));
  }} else if (state.zoomMode === 'fill') {{
    const availW = Math.max(320, stage.clientWidth - 24);
    const scale = availW / CONFIG.width;
    canvas.style.width = `${{Math.round(CONFIG.width * scale)}}px`;
    canvas.style.height = `${{Math.round(CONFIG.height * scale)}}px`;
    zoomLbl.textContent = `${{Math.round(scale * 100)}}%`;
    document.getElementById("zoom-slider").value = Math.min(300, Math.max(25, Math.round(scale * 100)));
  }} else if (state.zoomMode === '1to1') {{
    canvas.style.width = `${{CONFIG.width}}px`;
    canvas.style.height = `${{CONFIG.height}}px`;
    canvas.classList.add("pixelated");
    zoomLbl.textContent = "100%";
    document.getElementById("zoom-slider").value = 100;
  }} else {{
    const scale = state.customZoom / 100.0;
    canvas.style.width = `${{Math.round(CONFIG.width * scale)}}px`;
    canvas.style.height = `${{Math.round(CONFIG.height * scale)}}px`;
    if (scale >= 1.5) canvas.classList.add("pixelated");
    zoomLbl.textContent = `${{state.customZoom}}%`;
  }}
}}

function setStripMode(mode) {{
  state.stripMode = mode;
  document.getElementById("strip-mode-color").className = "btn" + (mode === 'color' ? " active" : "");
  document.getElementById("strip-mode-gray").className = "btn" + (mode === 'gray' ? " active" : "");
  requestRender();
}}

// Draw calibrated scale bar directly onto a full-resolution 2D canvas context
function drawScaleBarOnContext(ctx, width, height) {{
  const sb = state.scaleBar;
  if (!sb.enabled || sb.pxPerUm <= 0 || sb.um <= 0) return;

  const barPx = Math.round(sb.um * sb.pxPerUm);
  if (barPx < 2) return;

  const scaleMult = (sb.sizeScale || 100) / 100.0;
  const baseDim = Math.max(width, height);
  const marginX = Math.max(14, Math.round(width * 0.035));
  const marginY = Math.max(14, Math.round(height * 0.035));
  const barHeight = Math.max(4, Math.round(baseDim * 0.011 * scaleMult));
  const fontSize = Math.max(12, Math.round(baseDim * 0.024 * scaleMult));

  ctx.save();
  ctx.font = `bold ${{fontSize}}px -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif`;
  const label = `${{Number.isInteger(sb.um) ? sb.um : sb.um}} µm`;
  const textWidth = ctx.measureText(label).width;
  const boxWidth = Math.max(barPx, textWidth);

  let x1, y1;
  if (sb.pos === 'bl') {{
    x1 = marginX;
    y1 = height - marginY - barHeight;
  }} else if (sb.pos === 'tr') {{
    x1 = width - marginX - boxWidth;
    y1 = marginY + fontSize + 6;
  }} else if (sb.pos === 'tl') {{
    x1 = marginX;
    y1 = marginY + fontSize + 6;
  }} else {{
    x1 = width - marginX - boxWidth;
    y1 = height - marginY - barHeight;
  }}

  const padX = Math.max(6, Math.round(fontSize * 0.4));
  const padY = Math.max(5, Math.round(fontSize * 0.35));
  const bgX = x1 - padX;
  const bgY = y1 - fontSize - padY - 4;
  const bgW = boxWidth + padX * 2;
  const bgH = fontSize + barHeight + padY * 2 + 4;

  ctx.fillStyle = "rgba(0, 0, 0, 0.68)";
  ctx.fillRect(bgX, bgY, bgW, bgH);

  const barX = x1 + Math.round((boxWidth - barPx) / 2);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(barX, y1, barPx, barHeight);

  const textX = x1 + Math.round((boxWidth - textWidth) / 2);
  ctx.textBaseline = "bottom";
  ctx.fillText(label, textX, y1 - 3);
  ctx.restore();
}}

// Core full-resolution Presentation rendering loop (strictly isolated from rawAnalysisBuffers)
const mainCanvas = document.getElementById("main-canvas");
const mainCtx = mainCanvas.getContext("2d", {{ willReadFrequently: true }});
const mainImgData = mainCtx.createImageData(CONFIG.width, CONFIG.height);

function renderAll() {{
  const W = CONFIG.width;
  const H = CONFIG.height;
  const numPixels = W * H;
  const out = mainImgData.data;

  const activeCols = [];
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const cfg = state.channels[c];
    const [r, g, b] = hexToRgbNorm(cfg.hex);
    activeCols.push({{
      enabled: cfg.enabled,
      r255: r * 255.0,
      g255: g * 255.0,
      b255: b * 255.0,
      lut: lutNorm[c],
      plane: visPlanesU16[c]
    }});
  }}

  for (let i = 0; i < numPixels; i++) {{
    let rSum = 0.0, gSum = 0.0, bSum = 0.0;
    for (let c = 0; c < CONFIG.numChannels; c++) {{
      const ac = activeCols[c];
      if (!ac.enabled) continue;
      const norm = ac.lut[ac.plane[i]];
      if (norm > 0.0) {{
        rSum += norm * ac.r255;
        gSum += norm * ac.g255;
        bSum += norm * ac.b255;
      }}
    }}
    const idx = i << 2;
    out[idx]     = rSum >= 255.0 ? 255 : (rSum + 0.5) | 0;
    out[idx + 1] = gSum >= 255.0 ? 255 : (gSum + 0.5) | 0;
    out[idx + 2] = bSum >= 255.0 ? 255 : (bSum + 0.5) | 0;
    out[idx + 3] = 255;
  }}

  mainCtx.putImageData(mainImgData, 0, 0);
  drawScaleBarOnContext(mainCtx, W, H);

  renderIndividualChannelCanvases(activeCols, numPixels, W, H);
  renderHistogram(activeCols, numPixels);
}}

function renderIndividualChannelCanvases(activeCols, numPixels, W, H) {{
  const isGray = state.stripMode === 'gray';
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const cCanvas = document.getElementById(`thumb-canvas-${{c}}`);
    if (!cCanvas) continue;
    const ctx = cCanvas.getContext("2d");
    const imgData = ctx.createImageData(W, H);
    const data = imgData.data;
    const ac = activeCols[c];
    const lut = ac.lut;
    const plane = ac.plane;
    const rM = isGray ? 255.0 : ac.r255;
    const gM = isGray ? 255.0 : ac.g255;
    const bM = isGray ? 255.0 : ac.b255;

    for (let i = 0; i < numPixels; i++) {{
      const norm = lut[plane[i]];
      const idx = i << 2;
      data[idx]     = (norm * rM + 0.5) | 0;
      data[idx + 1] = (norm * gM + 0.5) | 0;
      data[idx + 2] = (norm * bM + 0.5) | 0;
      data[idx + 3] = 255;
    }}
    ctx.putImageData(imgData, 0, 0);
    drawScaleBarOnContext(ctx, W, H);
  }}
}}

function renderHistogram(activeCols, numPixels) {{
  const hCanvas = document.getElementById("hist-canvas");
  if (!hCanvas) return;
  const ctx = hCanvas.getContext("2d");
  const W = hCanvas.width;
  const H = hCanvas.height;

  ctx.fillStyle = "#0b0e14";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "#1f2630";
  ctx.lineWidth = 1;
  [0.25, 0.5, 0.75].forEach(f => {{
    const x = Math.round(W * f);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }});

  const numBins = 128;
  const step = Math.max(1, Math.floor(numPixels / 16384));
  const bins = new Uint32Array(numBins);

  for (let c = 0; c < CONFIG.numChannels; c++) {{
    const ac = activeCols[c];
    if (!ac.enabled) continue;
    bins.fill(0);
    const lut = ac.lut;
    const plane = ac.plane;
    for (let i = 0; i < numPixels; i += step) {{
      const norm = lut[plane[i]];
      const bIdx = Math.min(numBins - 1, (norm * numBins) | 0);
      bins[bIdx]++;
    }}

    let maxLog = 0.0;
    const logBins = new Float32Array(numBins);
    for (let b = 0; b < numBins; b++) {{
      const lv = Math.log1p(bins[b]);
      logBins[b] = lv;
      if (lv > maxLog) maxLog = lv;
    }}
    if (maxLog <= 0) continue;

    const hex = state.channels[c].hex;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let b = 0; b < numBins; b++) {{
      const x = (b / (numBins - 1)) * W;
      const y = H - (logBins[b] / maxLog) * (H - 10);
      ctx.lineTo(x, y);
    }}
    ctx.lineTo(W, H);
    ctx.closePath();
    ctx.fillStyle = hex + "44";
    ctx.fill();

    ctx.strokeStyle = hex;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.restore();
  }}
}}

function getCleanExportPrefix() {{
  const raw = (document.getElementById("export-filename").value || CONFIG.seriesName || "microscopy").trim();
  return raw.replace(/[^a-zA-Z0-9._-]+/g, "_");
}}

function triggerBlobDownload(blob, filename) {{
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {{
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }}, 250);
}}

// ============================================================================
// EXPORT 1: SAVE FOR ANALYSIS (RAW UNMODIFIED DATA — ImageJ Hyperstack via tifffile)
// Assembles a single multi-dimensional ImageJ hyperstack TIFF of shape (C, Y, X)
// using the exact binary header/IFD structure generated by Python's
// `tifffile.imwrite(..., imagej=True)`. Each selected channel is stored as a
// distinct, unmodified layer directly from `rawAnalysisBuffers` (never blended,
// flattened, or added together, and completely bypassing all visual adjustments).
// ============================================================================
function b64ToUint8Array(b64) {{
  if (!b64) return new Uint8Array(0);
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}}

function encodeRawAnalysisMinisblackTiff(selectedIndices, width, height) {{
  const numChannelsSelected = selectedIndices.length;
  const numPixels = width * height;

  // Determine target bit depth (16-bit if any selected channel is 16-bit, else 8-bit)
  let targetBits = 8;
  for (const idx of selectedIndices) {{
    if (rawAnalysisBuffers[idx].bits > 8) targetBits = 16;
  }}

  const tmplKey = `${{targetBits}}_${{numChannelsSelected}}`;
  const tmpl = CONFIG.imagejTemplates[tmplKey];
  const prefixBytes = b64ToUint8Array(tmpl.prefixB64);
  const suffixBytes = b64ToUint8Array(tmpl.suffixB64);

  // Stack selected channels independently along axis 0 -> shape (C, Y, X)
  const blobParts = [prefixBytes];
  for (let p = 0; p < numChannelsSelected; p++) {{
    const chIdx = selectedIndices[p];
    const rawObj = rawAnalysisBuffers[chIdx];
    if (rawObj.bits === targetBits) {{
      blobParts.push(rawObj.byteView);
    }} else {{
      // Promote 8-bit channel to 16-bit container without modifying intensity values
      const promoted = new Uint16Array(numPixels);
      const src = rawObj.typedView;
      for (let i = 0; i < numPixels; i++) promoted[i] = src[i];
      blobParts.push(new Uint8Array(promoted.buffer));
    }}
  }}
  if (suffixBytes.length > 0) {{
    blobParts.push(suffixBytes);
  }}

  return new Blob(blobParts, {{ type: "image/tiff" }});
}}

function downloadRawAnalysisTIFF() {{
  const selected = [];
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    if (state.analysisSelected[c]) selected.push(c);
  }}
  if (selected.length === 0) {{
    alert("Please select at least one channel in '1A. Save for Analysis (Raw Data)'.");
    return;
  }}
  const chTag = selected.map(i => `Ch${{i + 1}}`).join("-");
  const filename = `${{getCleanExportPrefix()}}_RAW_ANALYSIS_${{chTag}}_${{CONFIG.width}}x${{CONFIG.height}}.tif`;
  const blob = encodeRawAnalysisMinisblackTiff(selected, CONFIG.width, CONFIG.height);
  triggerBlobDownload(blob, filename);
}}

function downloadSingleChannelRawTIFF(c) {{
  const filename = `${{getCleanExportPrefix()}}_RAW_ANALYSIS_Ch${{c + 1}}_${{CONFIG.width}}x${{CONFIG.height}}.tif`;
  const blob = encodeRawAnalysisMinisblackTiff([c], CONFIG.width, CONFIG.height);
  triggerBlobDownload(blob, filename);
}}

// ============================================================================
// EXPORT 2: SAVE FOR PRESENTATION (8-BIT RGB VISUAL RENDER: PNG, JPEG, or TIFF)
// Exports the visual canvas exactly as it currently looks on the screen.
// ============================================================================
function encodePresentationRGB8Tiff(ctx, width, height) {{
  const imgData = ctx.getImageData(0, 0, width, height).data;
  const numPixels = width * height;
  const rgbByteLength = numPixels * 3;
  const headerSize = 8;
  const numTags = 12;
  const ifdSize = 2 + numTags * 12 + 4;
  const extraSize = 6 + 8 + 8;
  const totalSize = headerSize + rgbByteLength + (rgbByteLength % 2) + ifdSize + extraSize;

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);

  view.setUint16(0, 0x4949, true);
  view.setUint16(2, 42, true);
  const ifdOffset = headerSize + rgbByteLength + (rgbByteLength % 2);
  view.setUint32(4, ifdOffset, true);

  let dst = headerSize;
  for (let i = 0; i < numPixels; i++) {{
    const src = i << 2;
    bytes[dst++] = imgData[src];
    bytes[dst++] = imgData[src + 1];
    bytes[dst++] = imgData[src + 2];
  }}

  let pos = ifdOffset;
  view.setUint16(pos, numTags, true);
  pos += 2;

  const extraOffsetStart = ifdOffset + ifdSize;
  const bpsOffset = extraOffsetStart;
  const xResOffset = extraOffsetStart + 6;
  const yResOffset = extraOffsetStart + 14;

  function writeTag(tag, type, count, valueOrOffset) {{
    view.setUint16(pos, tag, true);
    view.setUint16(pos + 2, type, true);
    view.setUint32(pos + 4, count, true);
    if (type === 3 && count === 1) {{
      view.setUint16(pos + 8, valueOrOffset, true);
      view.setUint16(pos + 10, 0, true);
    }} else {{
      view.setUint32(pos + 8, valueOrOffset, true);
    }}
    pos += 12;
  }}

  writeTag(256, 4, 1, width);
  writeTag(257, 4, 1, height);
  writeTag(258, 3, 3, bpsOffset);      // BitsPerSample = (8, 8, 8)
  writeTag(259, 3, 1, 1);              // Uncompressed
  writeTag(262, 3, 1, 2);              // PhotometricInterpretation = 2 (RGB)
  writeTag(273, 4, 1, headerSize);
  writeTag(277, 3, 1, 3);              // SamplesPerPixel = 3
  writeTag(278, 4, 1, height);
  writeTag(279, 4, 1, rgbByteLength);
  writeTag(282, 5, 1, xResOffset);
  writeTag(283, 5, 1, yResOffset);
  writeTag(296, 3, 1, 1);

  view.setUint32(pos, 0, true);

  view.setUint16(bpsOffset, 8, true);
  view.setUint16(bpsOffset + 2, 8, true);
  view.setUint16(bpsOffset + 4, 8, true);

  const resNum = Math.max(1, Math.round((state.scaleBar.pxPerUm || 1) * 10000));
  view.setUint32(xResOffset, resNum, true);
  view.setUint32(xResOffset + 4, 10000, true);
  view.setUint32(yResOffset, resNum, true);
  view.setUint32(yResOffset + 4, 10000, true);

  return new Blob([buffer], {{ type: "image/tiff" }});
}}

function exportCanvasInPresentationFormat(canvas, ctx, baseName) {{
  const fmt = document.getElementById("pres-format-select").value || "PNG";
  if (fmt === "TIFF") {{
    const filename = `${{baseName}}_8bit_RGB.tif`;
    const blob = encodePresentationRGB8Tiff(ctx, CONFIG.width, CONFIG.height);
    triggerBlobDownload(blob, filename);
  }} else if (fmt === "JPEG") {{
    const filename = `${{baseName}}.jpg`;
    canvas.toBlob(blob => {{
      if (blob) triggerBlobDownload(blob, filename);
    }}, "image/jpeg", 0.96);
  }} else {{
    const filename = `${{baseName}}.png`;
    canvas.toBlob(blob => {{
      if (blob) triggerBlobDownload(blob, filename);
    }}, "image/png");
  }}
}}

function downloadPresentationImage() {{
  const baseName = `${{getCleanExportPrefix()}}_PRESENTATION_${{CONFIG.width}}x${{CONFIG.height}}`;
  exportCanvasInPresentationFormat(mainCanvas, mainCtx, baseName);
}}

function downloadSingleChannelPresentation(c) {{
  const cCanvas = document.getElementById(`thumb-canvas-${{c}}`);
  if (!cCanvas) return;
  const ctx = cCanvas.getContext("2d");
  const presetClean = state.channels[c].preset.replace(/[^a-zA-Z0-9]+/g, "_");
  const baseName = `${{getCleanExportPrefix()}}_PRESENTATION_Ch${{c + 1}}_${{presetClean}}_${{CONFIG.width}}x${{CONFIG.height}}`;
  exportCanvasInPresentationFormat(cCanvas, ctx, baseName);
}}

function downloadAllSelectedChannelsPresentation() {{
  downloadPresentationImage();
  let delay = 250;
  for (let c = 0; c < CONFIG.numChannels; c++) {{
    if (state.channels[c].enabled) {{
      setTimeout(() => downloadSingleChannelPresentation(c), delay);
      delay += 250;
    }}
  }}
}}

// Initialize everything on startup
loadSavedState();
for (let c = 0; c < CONFIG.numChannels; c++) {{
  updateChannelLUT(c);
}}
buildUI();
updateUndoRedoButtons();
applyCanvasDisplaySize();
renderAll();
window.addEventListener("resize", () => {{
  if (state.zoomMode === 'fit' || state.zoomMode === 'fill') applyCanvasDisplaySize();
}});
window.addEventListener("pointerup", commitContinuousEdit);
window.addEventListener("mouseup", commitContinuousEdit);
window.addEventListener("change", commitContinuousEdit);
window.addEventListener("keydown", handleGlobalKeyDown);
try {{
  if (window.parent && window.parent.document) {{
    window.parent.document.addEventListener("keydown", handleGlobalKeyDown);
  }}
}} catch (e) {{}}
</script>
</body>
</html>
"""
    return html
