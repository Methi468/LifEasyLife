"""
Streamlit Web Application: Leica .lif Microscopy Viewer & Real-Time Studio.

Strictly isolates two export pipelines:
1. 'Save for Analysis' (Raw Data):
   - Uses `tifffile` (`export_raw_analysis_tiff`) on immutable raw integer arrays
     (`uint16` or `uint8`) directly from `readlif`.
   - Completely bypasses all visual LUTs, brightness, contrast, clipping, and pseudocolors.
2. 'Save for Presentation' (Visual Render):
   - Exports the 8-bit RGB image exactly as it currently looks on the screen
     (with all Lightroom-style tonal adjustments, clipping, pseudocolors, and scale bar baked in)
     in the user's choice of PNG, JPEG, or TIFF (8-bit RGB).
"""

import hashlib
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from image_processing import (
    compute_z_projection,
    export_raw_analysis_tiff,
    extract_raw_analysis_plane,
)
from lif_loader import (
    READLIF_AVAILABLE,
    SeriesMetadata,
    generate_synthetic_demo_series,
    load_series_channel_stacks,
    parse_lif_series_list,
)
from realtime_studio import build_realtime_studio_html


st.set_page_config(
    page_title="LifEasyLife — Leica .lif Microscopy Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Remove the white sticky top header bar and Deploy button so it never covers the app */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2rem !important;
        pointer-events: none;
    }
    header[data-testid="stHeader"] * {
        pointer-events: auto;
    }
    [data-testid="stAppDeployButton"],
    .stAppDeployButton,
    .stDeployButton,
    [data-testid="stToolbar"] {
        display: none !important;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
    }
    iframe {
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _get_cached_temp_lif_path(uploaded_file: Any) -> Tuple[str, str]:
    """
    Stream an uploaded .lif file (up to 2 GB+) to a deterministic temporary file path
    for `readlif` without duplicating the full buffer in memory on every rerun.
    """
    uploaded_file.seek(0)
    head_bytes = uploaded_file.read(65536)
    uploaded_file.seek(0)

    file_size = getattr(uploaded_file, "size", len(head_bytes))
    safe_name = os.path.basename(getattr(uploaded_file, "name", "upload.lif"))
    fingerprint_src = head_bytes + f":{file_size}:{safe_name}".encode("utf-8")
    digest = hashlib.sha256(fingerprint_src).hexdigest()[:16]
    dataset_id = f"lif_{digest[:10]}"

    temp_dir = os.path.join(tempfile.gettempdir(), "leica_lif_streamlit_cache")
    os.makedirs(temp_dir, exist_ok=True)
    target_path = os.path.join(temp_dir, f"{digest}_{safe_name}")

    if not os.path.exists(target_path) or os.path.getsize(target_path) != file_size:
        uploaded_file.seek(0)
        with open(target_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f, length=16 * 1024 * 1024)
        uploaded_file.seek(0)

    return target_path, dataset_id


@st.cache_data(show_spinner="Parsing Leica .lif series metadata with readlif...")
def cached_parse_series(file_path: str) -> List[SeriesMetadata]:
    """Cache the parsed series metadata from a .lif file path."""
    return parse_lif_series_list(file_path)


@st.cache_data(show_spinner="Extracting unmodified raw channel Z-stacks from .lif series...")
def cached_load_stacks(
    file_path: str, series_index: int, t_index: int, m_index: int
) -> List[np.ndarray]:
    """
    Cache the immutable 3D (Z, Y, X) raw integer (`uint8` or `uint16`) channel stacks
    extracted directly from `readlif`.
    """
    return load_series_channel_stacks(file_path, series_index, t_index, m_index)


@st.cache_data(show_spinner=False)
def cached_demo_dataset() -> Tuple[List[SeriesMetadata], Dict[int, List[np.ndarray]]]:
    """Cache the built-in synthetic multi-channel Z-stack demo dataset."""
    return generate_synthetic_demo_series()


def main() -> None:
    if not READLIF_AVAILABLE:
        st.error(
            "The `readlif` library is not installed in the current Python environment. "
            "Run `pip install -r requirements.txt` to enable `.lif` file parsing."
        )

    # --- SIDEBAR: FILE UPLOAD, SERIES SELECTION & Z-STACK / MIP ---
    with st.sidebar:
        st.markdown("## 🔬 LifEasy")
        st.caption("Leica `.lif` Microscopy Viewer & Real-Time Studio")
        st.header("📂 1. Data Source")
        uploaded_file = st.file_uploader(
            "Upload Leica `.lif` File",
            type=["lif"],
            help="Select a Leica Image Format (.lif) file (up to 2 GB) to parse series, channels, and Z-stacks via readlif.",
        )

        use_demo = False
        if uploaded_file is None:
            use_demo = st.toggle(
                "Load Synthetic Confocal Z-Stack Demo",
                value=True,
                help="Explore the real-time Color Adjustment sliders, Alexa 568 / 647 LUTs, 10x/40x/60x scale bar, and MIP right away.",
            )

    if uploaded_file is None and not use_demo:
        st.info(
            "👈 Upload a Leica **`.lif`** file in the sidebar (or enable the built-in Synthetic Confocal Z-Stack Demo) to begin."
        )
        return

    # Load series metadata & immutable raw channel Z-stacks
    if uploaded_file is not None:
        try:
            temp_lif_path, dataset_id = _get_cached_temp_lif_path(uploaded_file)
            series_list = cached_parse_series(temp_lif_path)
        except Exception as exc:
            st.error(f"Failed to parse `.lif` file with `readlif`: {exc}")
            return

        if not series_list:
            st.warning("No image series were found in the uploaded `.lif` file.")
            return
    else:
        dataset_id = "demo"
        temp_lif_path = ""
        series_list, demo_stacks_map = cached_demo_dataset()

    with st.sidebar:
        st.divider()
        st.header("🧬 2. Series & Dimensions")
        series_options = list(range(len(series_list)))
        selected_series_idx = st.selectbox(
            "Select Series",
            options=series_options,
            format_func=lambda idx: series_list[idx].display_label,
        )
        active_meta = series_list[selected_series_idx]

        t_index = 0
        if active_meta.dim_t > 1:
            t_index = (
                st.slider(
                    "Timepoint (T)",
                    min_value=1,
                    max_value=active_meta.dim_t,
                    value=1,
                    step=1,
                )
                - 1
            )

        m_index = 0
        if active_meta.dim_m > 1:
            m_index = (
                st.slider(
                    "Mosaic Tile (M)",
                    min_value=1,
                    max_value=active_meta.dim_m,
                    value=1,
                    step=1,
                )
                - 1
            )

        st.divider()
        st.header("🥞 3. Z-Stack & Projection")
        is_z_stack = active_meta.is_z_stack

        use_mip = st.toggle(
            "Maximum Intensity Projection (MIP)",
            value=is_z_stack,
            disabled=not is_z_stack,
            help=(
                "Project the maximum pixel intensity across Z-slices for each channel."
                if is_z_stack
                else "Current series has only 1 Z-plane (Z=1)."
            ),
        )

        projection_mode = "Maximum Intensity (MIP)"
        z_range: Optional[Tuple[int, int]] = None
        z_slice_idx = 0

        if is_z_stack:
            if use_mip:
                projection_mode = st.selectbox(
                    "Projection Method",
                    options=[
                        "Maximum Intensity (MIP)",
                        "Average Intensity",
                        "Sum Intensity",
                        "Minimum Intensity",
                    ],
                    index=0,
                )
                z_sub = st.slider(
                    "Z-Slices Included in Projection",
                    min_value=1,
                    max_value=active_meta.dim_z,
                    value=(1, active_meta.dim_z),
                    step=1,
                    help="Restrict the Z-projection to a specific sub-stack range if desired.",
                )
                z_range = (z_sub[0] - 1, z_sub[1] - 1)
            else:
                z_one_based = st.slider(
                    "Active Z-Slice",
                    min_value=1,
                    max_value=active_meta.dim_z,
                    value=(active_meta.dim_z // 2) + 1,
                    step=1,
                )
                z_slice_idx = z_one_based - 1
                if active_meta.scale_z and active_meta.scale_z > 0:
                    z_um = z_slice_idx / active_meta.scale_z
                    st.caption(
                        f"Focal Depth: **{z_um:.2f} µm** (Slice {z_one_based}/{active_meta.dim_z})"
                    )
        else:
            st.caption(
                "Single Z-plane series (`Z = 1`). Select a Z-stack series to slice or project across Z."
            )

    # Extract immutable raw 3D stacks directly from `readlif`
    if uploaded_file is not None:
        try:
            raw_channel_stacks = cached_load_stacks(
                temp_lif_path, selected_series_idx, t_index, m_index
            )
        except Exception as exc:
            st.error(
                f"Error extracting frames from series {selected_series_idx + 1}: {exc}"
            )
            return
    else:
        raw_channel_stacks = demo_stacks_map[selected_series_idx]

    # ==========================================================================
    # PIPELINE 1 (ISOLATED RAW DATA FOR QUANTITATIVE ANALYSIS):
    # Strictly preserves original integer dtype (`uint8` or `uint16`) from `readlif`
    # with `writeable = False`. Never touched by LUTs, brightness, or contrast.
    # ==========================================================================
    raw_analysis_planes_2d: List[np.ndarray] = [
        extract_raw_analysis_plane(
            stack_3d,
            use_mip=use_mip,
            z_index=z_slice_idx,
            z_range=z_range,
        )
        for stack_3d in raw_channel_stacks
    ]

    # ==========================================================================
    # PIPELINE 2 (VISUAL PRESENTATION RENDERING):
    # Separate float32 copies used exclusively for the visual display & 8-bit RGB
    # presentation export.
    # ==========================================================================
    presentation_planes_2d: List[np.ndarray] = [
        compute_z_projection(
            stack_3d,
            use_mip=use_mip,
            z_index=z_slice_idx,
            z_range=z_range,
            projection_mode=projection_mode,
        )
        for stack_3d in raw_channel_stacks
    ]

    # --- SIDEBAR SECTION 4: SAVE FOR ANALYSIS (RAW DATA VIA `tifffile`) ---
    with st.sidebar:
        st.divider()
        st.header("🔬 4. Save for Analysis (Raw TIFF)")
        st.caption(
            "Exports unmodified raw data directly from `readlif` using **`tifffile`**. "
            "Completely bypasses all LUTs, brightness, contrast, and pseudocolors."
        )
        ch_indices = list(range(len(raw_channel_stacks)))
        selected_raw_channels = st.multiselect(
            "Select Channels to Include / Merge",
            options=ch_indices,
            default=ch_indices,
            format_func=lambda idx: (
                f"{active_meta.channel_names[idx]} "
                f"({active_meta.bit_depths[idx] if idx < len(active_meta.bit_depths) else 8}-bit)"
            ),
        )

        export_full_z = False
        if is_z_stack:
            raw_scope = st.radio(
                "Raw Data Scope",
                options=[
                    "Active 2D Plane (Current Z-Slice or MIP)",
                    f"Full 3D Z-Stack (All {active_meta.dim_z} Z-Slices)",
                ],
                index=0,
            )
            export_full_z = raw_scope.startswith("Full 3D")

        if selected_raw_channels:
            source_arrays = (
                raw_channel_stacks if export_full_z else raw_analysis_planes_2d
            )
            raw_tiff_bytes = export_raw_analysis_tiff(
                raw_arrays_by_channel=source_arrays,
                selected_channel_indices=selected_raw_channels,
                channel_names=active_meta.channel_names,
                scale_x=active_meta.scale_x,
                scale_y=active_meta.scale_y,
                scale_z=active_meta.scale_z,
            )
            ch_tag = "-".join(f"Ch{idx + 1}" for idx in selected_raw_channels)
            scope_tag = (
                f"3D_Z{active_meta.dim_z}"
                if export_full_z
                else ("MIP" if (is_z_stack and use_mip) else f"Z{z_slice_idx + 1}")
            )
            clean_series = active_meta.name.replace("/", "_")
            raw_filename = f"{clean_series}_RAW_ANALYSIS_{ch_tag}_{scope_tag}.tif"

            sample_dtype = source_arrays[selected_raw_channels[0]].dtype
            st.caption(
                f"✅ **Unmodified `{sample_dtype}`** • {len(selected_raw_channels)} ch • "
                f"`{active_meta.dim_x}×{active_meta.dim_y}` px"
            )
            st.download_button(
                "⬇️ Save for Analysis (Raw TIFF via tifffile)",
                data=raw_tiff_bytes,
                file_name=raw_filename,
                mime="image/tiff",
                width="stretch",
            )
        else:
            st.warning("Select at least one channel above to export Raw Analysis TIFF.")

    mode_label = (
        f"{projection_mode} (Z {z_range[0] + 1}–{z_range[1] + 1} / {active_meta.dim_z})"
        if (is_z_stack and use_mip and z_range is not None)
        else (
            f"Z-Slice {z_slice_idx + 1} / {active_meta.dim_z}"
            if is_z_stack
            else "Single Focal Plane (Z=1)"
        )
    )

    # Render the full-resolution Real-Time HTML5 Studio
    studio_html = build_realtime_studio_html(
        dataset_id=dataset_id,
        meta=active_meta,
        presentation_planes_2d=presentation_planes_2d,
        mode_label=mode_label,
        raw_analysis_planes_2d=raw_analysis_planes_2d,
    )
    components.html(studio_html, height=1220, scrolling=True)


if __name__ == "__main__":
    main()
