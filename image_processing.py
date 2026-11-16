"""
Image processing and colorimetric utilities for microscopy channels.

Provides:
- Fluorophore / pseudocolor presets (including Alexa Fluor dyes like Alexa 568, Alexa 647)
- Maximum Intensity Projection (MIP) across Z-stacks
- Lightroom-style per-channel tonal adjustments (Black/White points, Brightness, Contrast, Gamma, Exposure)
- Multi-channel additive fluorescence composite blending
- Histogram and scale bar rendering utilities
"""

from dataclasses import dataclass
import io
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from PIL import Image, ImageDraw
import tifffile


@dataclass(frozen=True)
class FluorophorePreset:
    """Represents a microscopy fluorophore or pseudocolor preset."""
    name: str
    hex_color: str
    ex_nm: Optional[int] = None
    em_nm: Optional[int] = None
    description: str = ""

    @property
    def rgb_normalized(self) -> Tuple[float, float, float]:
        """Return (R, G, B) tuple in [0.0, 1.0]."""
        return hex_to_rgb_norm(self.hex_color)


# Curated catalog of common fluorophores and microscopy LUT colors
FLUOROPHORE_PRESETS: Dict[str, FluorophorePreset] = {
    "Alexa 405 / DAPI (Blue)": FluorophorePreset(
        name="Alexa 405 / DAPI (Blue)",
        hex_color="#3B82F6",
        ex_nm=401,
        em_nm=421,
        description="Violet excitation, blue emission (nuclear counterstain / Alexa Fluor 405)",
    ),
    "Alexa 488 / GFP (Green)": FluorophorePreset(
        name="Alexa 488 / GFP (Green)",
        hex_color="#00FF44",
        ex_nm=490,
        em_nm=525,
        description="Cyan-blue excitation, bright green emission (Alexa Fluor 488, FITC, EGFP)",
    ),
    "Alexa 532 (Yellow-Green)": FluorophorePreset(
        name="Alexa 532 (Yellow-Green)",
        hex_color="#CCFF00",
        ex_nm=532,
        em_nm=554,
        description="Green-yellow emission fluorophore (Alexa Fluor 532)",
    ),
    "Alexa 546 / Cy3 (Yellow-Orange)": FluorophorePreset(
        name="Alexa 546 / Cy3 (Yellow-Orange)",
        hex_color="#FFA500",
        ex_nm=556,
        em_nm=573,
        description="Orange emission dye (Alexa Fluor 546, TRITC, Cy3)",
    ),
    "Alexa 568 (Orange-Red)": FluorophorePreset(
        name="Alexa 568 (Orange-Red)",
        hex_color="#FF4500",
        ex_nm=578,
        em_nm=603,
        description="Yellow-orange excitation, orange-red emission at 603 nm (Alexa Fluor 568)",
    ),
    "Alexa 594 / mCherry (Red)": FluorophorePreset(
        name="Alexa 594 / mCherry (Red)",
        hex_color="#FF1A1A",
        ex_nm=590,
        em_nm=617,
        description="Orange-red excitation, deep red emission at 617 nm (Alexa Fluor 594, Texas Red)",
    ),
    "Alexa 647 (Magenta)": FluorophorePreset(
        name="Alexa 647 (Magenta)",
        hex_color="#FF00FF",
        ex_nm=650,
        em_nm=665,
        description="Far-red emission at 665 nm mapped to high-visibility Magenta (Alexa Fluor 647, Cy5)",
    ),
    "Alexa 647 (Far-Red)": FluorophorePreset(
        name="Alexa 647 (Far-Red)",
        hex_color="#E60033",
        ex_nm=650,
        em_nm=665,
        description="Far-red emission at 665 nm mapped to spectral Crimson/Far-Red (Alexa Fluor 647)",
    ),
    "Alexa 680 (Deep Crimson)": FluorophorePreset(
        name="Alexa 680 (Deep Crimson)",
        hex_color="#B30047",
        ex_nm=679,
        em_nm=702,
        description="Near-IR emission at 702 nm (Alexa Fluor 680)",
    ),
    "Cyan (CFP / Aqua)": FluorophorePreset(
        name="Cyan (CFP / Aqua)",
        hex_color="#00FFFF",
        ex_nm=435,
        em_nm=477,
        description="Colorblind-safe Cyan LUT",
    ),
    "Yellow (YFP / Gold)": FluorophorePreset(
        name="Yellow (YFP / Gold)",
        hex_color="#FFFF00",
        ex_nm=514,
        em_nm=527,
        description="Bright Yellow LUT (EYFP, Venus)",
    ),
    "Grayscale / White": FluorophorePreset(
        name="Grayscale / White",
        hex_color="#FFFFFF",
        ex_nm=None,
        em_nm=None,
        description="Neutral monochrome display (DIC / Brightfield / Single-channel inspection)",
    ),
    "Custom Color": FluorophorePreset(
        name="Custom Color",
        hex_color="#00E5FF",
        ex_nm=None,
        em_nm=None,
        description="User-defined RGB hex pseudocolor",
    ),
}

# Default preset assignments for up to 6 channels
DEFAULT_CHANNEL_PRESETS: List[str] = [
    "Alexa 488 / GFP (Green)",
    "Alexa 568 (Orange-Red)",
    "Alexa 647 (Magenta)",
    "Alexa 405 / DAPI (Blue)",
    "Cyan (CFP / Aqua)",
    "Yellow (YFP / Gold)",
]


@dataclass
class ChannelSettings:
    """Lightroom-style adjustment settings for a single microscopy channel."""
    enabled: bool = True
    preset_name: str = "Alexa 488 / GFP (Green)"
    custom_hex: str = "#00FF44"
    exposure: float = 0.0        # EV stops in [-3.0, +3.0]
    brightness: float = 0.0      # [-100.0, +100.0]
    contrast: float = 0.0        # [-100.0, +100.0]
    gamma: float = 1.0           # [0.10, 3.00]
    black_point: float = 0.0     # Raw intensity black clipping level
    white_point: float = 255.0   # Raw intensity white clipping level
    invert: bool = False


def hex_to_rgb_norm(hex_color: str) -> Tuple[float, float, float]:
    """Convert a hex color string (e.g. '#FF4500') to normalized floats (r, g, b) in [0, 1]."""
    clean = hex_color.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(c * 2 for c in clean)
    if len(clean) != 6:
        return (1.0, 1.0, 1.0)
    r = int(clean[0:2], 16) / 255.0
    g = int(clean[2:4], 16) / 255.0
    b = int(clean[4:6], 16) / 255.0
    return (r, g, b)


def compute_z_projection(
    z_stack: np.ndarray,
    use_mip: bool = True,
    z_index: int = 0,
    z_range: Optional[Tuple[int, int]] = None,
    projection_mode: str = "Maximum Intensity (MIP)",
) -> np.ndarray:
    """
    Extract a 2D plane from a 3D Z-stack array of shape (Z, Y, X).

    Parameters
    ----------
    z_stack : np.ndarray
        Array of shape (Z, Y, X) or (Y, X).
    use_mip : bool
        If True and Z > 1, compute a Z-projection across `z_range`.
        If False, extract the single slice at `z_index`.
    z_index : int
        0-indexed Z slice when `use_mip` is False.
    z_range : Optional[Tuple[int, int]]
        Inclusive (z_start, z_end) slice indices for projection.
    projection_mode : str
        One of 'Maximum Intensity (MIP)', 'Average Intensity', 'Sum Intensity', 'Minimum Intensity'.

    Returns
    -------
    np.ndarray
        2D float32 plane of shape (Y, X).
    """
    if z_stack.ndim == 2:
        return z_stack.astype(np.float32, copy=False)

    num_z = z_stack.shape[0]
    if num_z == 1 or not use_mip:
        clamped_z = int(np.clip(z_index, 0, num_z - 1))
        return z_stack[clamped_z].astype(np.float32, copy=False)

    if z_range is not None:
        z_start = int(np.clip(z_range[0], 0, num_z - 1))
        z_end = int(np.clip(z_range[1], z_start, num_z - 1))
        sub_stack = z_stack[z_start : z_end + 1]
    else:
        sub_stack = z_stack

    if projection_mode == "Average Intensity":
        return np.mean(sub_stack, axis=0, dtype=np.float32)
    elif projection_mode == "Sum Intensity":
        return np.sum(sub_stack, axis=0, dtype=np.float32)
    elif projection_mode == "Minimum Intensity":
        return np.min(sub_stack, axis=0).astype(np.float32, copy=False)
    else:
        # Default: Maximum Intensity Projection (MIP)
        return np.max(sub_stack, axis=0).astype(np.float32, copy=False)


def compute_auto_levels(
    plane: np.ndarray, low_percentile: float = 0.2, high_percentile: float = 99.8
) -> Tuple[float, float]:
    """Calculate percentile-based black and white clipping points for auto-contrast."""
    if plane.size == 0:
        return (0.0, 255.0)
    low, high = np.percentile(plane, [low_percentile, high_percentile])
    if high <= low:
        high = float(np.max(plane))
        low = float(np.min(plane))
    if high <= low:
        high = low + 1.0
    return (float(low), float(high))


def apply_lightroom_adjustments(
    plane: np.ndarray,
    settings: ChannelSettings,
) -> np.ndarray:
    """
    Apply Lightroom-style tonal adjustments to a 2D channel array.

    Pipeline:
    1. Black Point / White Point window normalization to [0, 1]
    2. Exposure compensation in EV stops: I * (2 ** exposure)
    3. Brightness offset and Contrast S-curve / slope around midpoint 0.5
    4. Gamma correction: I ** (1 / gamma)
    5. Optional inversion

    Returns
    -------
    np.ndarray
        2D float32 array normalized and clamped to [0.0, 1.0].
    """
    arr = plane.astype(np.float32, copy=False)

    # 1. Black/White point windowing
    denom = max(settings.white_point - settings.black_point, 1e-6)
    norm = (arr - settings.black_point) / denom
    norm = np.clip(norm, 0.0, 1.0)

    # 2. Exposure in photographic stops (EV)
    if abs(settings.exposure) > 1e-4:
        norm = norm * (2.0 ** float(settings.exposure))

    # 3. Contrast and Brightness (Lightroom-style)
    # Contrast in [-100, 100] maps to a tangent/slope factor around midpoint 0.5
    # Brightness in [-100, 100] shifts tonal values smoothly
    if abs(settings.contrast) > 1e-4 or abs(settings.brightness) > 1e-4:
        c_norm = np.clip(settings.contrast / 100.0, -0.98, 0.98)
        # Photographic contrast factor using tangent mapping
        contrast_factor = float(np.tan((c_norm + 1.0) * (np.pi / 4.0)))
        b_offset = float(settings.brightness / 100.0) * 0.5

        # Apply brightness pre-shift + midpoint contrast expansion
        norm = contrast_factor * (norm - 0.5 + b_offset) + 0.5
        norm = np.clip(norm, 0.0, 1.0)
    else:
        norm = np.clip(norm, 0.0, 1.0)

    # 4. Gamma adjustment
    gamma_val = max(float(settings.gamma), 0.05)
    if abs(gamma_val - 1.0) > 1e-4:
        norm = np.power(norm, 1.0 / gamma_val)

    # 5. Optional invert
    if settings.invert:
        norm = 1.0 - norm

    return np.clip(norm, 0.0, 1.0)


def colorize_channel(
    adjusted_plane: np.ndarray,
    hex_color: str,
) -> np.ndarray:
    """
    Map a 2D normalized float32 plane in [0, 1] to an RGB float32 image (Y, X, 3) in [0, 1].
    """
    r, g, b = hex_to_rgb_norm(hex_color)
    rgb = np.empty((*adjusted_plane.shape, 3), dtype=np.float32)
    rgb[..., 0] = adjusted_plane * r
    rgb[..., 1] = adjusted_plane * g
    rgb[..., 2] = adjusted_plane * b
    return rgb


def blend_channels(
    adjusted_planes: List[np.ndarray],
    channel_settings: List[ChannelSettings],
) -> np.ndarray:
    """
    Blend multiple adjusted 2D planes into a single additive RGB uint8 composite image (Y, X, 3).

    Additive blending mimics fluorescence microscopy emission superposition.
    """
    if not adjusted_planes:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    h, w = adjusted_planes[0].shape
    composite = np.zeros((h, w, 3), dtype=np.float32)

    for plane, cfg in zip(adjusted_planes, channel_settings):
        if not cfg.enabled:
            continue
        rgb_plane = colorize_channel(plane, cfg.custom_hex)
        composite += rgb_plane

    composite = np.clip(composite, 0.0, 1.0)
    return (composite * 255.0 + 0.5).astype(np.uint8)


def draw_scale_bar(
    rgb_uint8: np.ndarray,
    scale_px_per_um: Optional[float],
    bar_um: Optional[float] = None,
) -> np.ndarray:
    """
    Draw a calibrated microscopy scale bar in the bottom-right corner of an RGB uint8 image.
    """
    if scale_px_per_um is None or scale_px_per_um <= 0:
        return rgb_uint8

    h, w, _ = rgb_uint8.shape
    # Pick a clean round scale bar length in micrometers if not specified
    fov_um = w / scale_px_per_um
    if bar_um is None:
        target_um = fov_um * 0.18
        nice_steps = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]
        bar_um = float(min(nice_steps, key=lambda x: abs(x - target_um)))

    bar_px = int(round(bar_um * scale_px_per_um))
    if bar_px < 10 or bar_px > w - 20:
        return rgb_uint8

    pil_img = Image.fromarray(rgb_uint8)
    draw = ImageDraw.Draw(pil_img)

    margin_x = max(16, int(w * 0.04))
    margin_y = max(16, int(h * 0.04))
    bar_height = max(4, int(h * 0.012))

    x1 = w - margin_x - bar_px
    x2 = w - margin_x
    y2 = h - margin_y
    y1 = y2 - bar_height

    # Subtle dark backing box for readability
    pad = 6
    draw.rectangle(
        [x1 - pad, y1 - 18, x2 + pad, y2 + pad],
        fill=(0, 0, 0),
    )
    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
    label = f"{int(bar_um) if bar_um.is_integer() else bar_um} µm"
    draw.text((x1, y1 - 15), label, fill=(255, 255, 255))

    return np.asarray(pil_img, dtype=np.uint8)


# ==============================================================================
# STRICTLY ISOLATED EXPORT PIPELINES:
# 1. SAVE FOR ANALYSIS (RAW DATA via `tifffile`, zero LUT/tonal/color changes)
# 2. SAVE FOR PRESENTATION (8-bit RGB Visual Render: PNG, JPEG, or 8-bit TIFF)
# ==============================================================================


def extract_raw_analysis_plane(
    raw_z_stack: np.ndarray,
    use_mip: bool = True,
    z_index: int = 0,
    z_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Extract a 2D plane from a raw 3D Z-stack for quantitative analysis, strictly
    preserving the original native integer dtype (`uint8` or `uint16`) from `readlif`.

    No float conversion, normalization, LUT, brightness, contrast, clipping, or
    pseudocolor is ever applied. Returns a read-only copy (`writeable = False`).
    """
    orig_dtype = raw_z_stack.dtype
    if raw_z_stack.ndim == 2:
        out = np.array(raw_z_stack, dtype=orig_dtype, copy=True)
        out.flags.writeable = False
        return out

    num_z = raw_z_stack.shape[0]
    if num_z == 1 or not use_mip:
        clamped_z = int(np.clip(z_index, 0, num_z - 1))
        out = np.array(raw_z_stack[clamped_z], dtype=orig_dtype, copy=True)
        out.flags.writeable = False
        return out

    if z_range is not None:
        z_start = int(np.clip(z_range[0], 0, num_z - 1))
        z_end = int(np.clip(z_range[1], z_start, num_z - 1))
        sub_stack = raw_z_stack[z_start : z_end + 1]
    else:
        sub_stack = raw_z_stack

    # Maximum Intensity Projection across Z preserves exact native integer dtype
    out = np.max(sub_stack, axis=0).astype(orig_dtype, copy=True)
    out.flags.writeable = False
    return out


def export_raw_analysis_tiff(
    raw_arrays_by_channel: Sequence[np.ndarray],
    selected_channel_indices: Sequence[int],
    channel_names: Optional[Sequence[str]] = None,
    scale_x: Optional[float] = None,
    scale_y: Optional[float] = None,
    scale_z: Optional[float] = None,
) -> bytes:
    """
    Save selected channels as a single multi-dimensional ImageJ hyperstack TIFF
    file using `tifffile.imwrite(..., imagej=True)`.

    - Strictly preserves original intensity values and native integer dtype (`uint16` / `uint8`)
      directly from the `readlif` parser.
    - Formats multi-channel 2D data as `(C, Y, X)` (`axes='CYX'`) and multi-channel 3D Z-stacks
      as `(Z, C, Y, X)` (`axes='ZCYX'`) so channels remain completely independent layers.
    - NEVER mathematically blends, flattens, or adds channel arrays together.
    - Completely bypasses all visual LUTs, brightness, contrast, gamma, clipping,
      and pseudocolor adjustments.
    """
    if not selected_channel_indices:
        raise ValueError("At least one channel must be selected for Raw Analysis export.")

    picked = [raw_arrays_by_channel[idx] for idx in selected_channel_indices]
    # Promote to the widest integer dtype among selected channels (uint8 or uint16)
    target_dtype = np.result_type(*[arr.dtype for arr in picked])
    if not np.issubdtype(target_dtype, np.integer):
        target_dtype = np.uint16

    picked_typed = [np.asarray(arr, dtype=target_dtype) for arr in picked]

    selected_labels = [
        (
            channel_names[idx]
            if (channel_names and idx < len(channel_names))
            else f"Channel {idx + 1}"
        )
        for idx in selected_channel_indices
    ]

    if picked_typed[0].ndim == 2:
        if len(picked_typed) == 1:
            raw_volume = picked_typed[0]  # shape: (Y, X)
            axes = "YX"
            ij_labels = selected_labels
        else:
            raw_volume = np.stack(picked_typed, axis=0)  # shape: (C, Y, X)
            axes = "CYX"
            ij_labels = selected_labels
    elif picked_typed[0].ndim == 3:
        num_z = picked_typed[0].shape[0]
        if len(picked_typed) == 1:
            raw_volume = picked_typed[0]  # shape: (Z, Y, X)
            axes = "ZYX"
            ij_labels = [f"Z{z + 1}_{selected_labels[0]}" for z in range(num_z)]
        else:
            raw_volume = np.stack(picked_typed, axis=1)  # shape: (Z, C, Y, X)
            axes = "ZCYX"
            ij_labels = [
                f"Z{z + 1}_{lbl}" for z in range(num_z) for lbl in selected_labels
            ]
    else:
        raise ValueError(f"Unsupported array dimensionality: {picked_typed[0].ndim}")

    metadata: Dict[str, object] = {
        "axes": axes,
        "mode": "composite",
        "Labels": ij_labels,
    }
    if scale_z and scale_z > 0 and "Z" in axes:
        metadata["spacing"] = 1.0 / float(scale_z)

    resolution = None
    if scale_x and scale_x > 0:
        sy = scale_y if (scale_y and scale_y > 0) else scale_x
        resolution = (float(scale_x), float(sy))
        metadata["unit"] = "um"

    buf = io.BytesIO()
    if resolution is not None:
        tifffile.imwrite(
            buf,
            raw_volume,
            imagej=True,
            photometric="minisblack",
            resolution=resolution,
            metadata=metadata,
        )
    else:
        tifffile.imwrite(
            buf,
            raw_volume,
            imagej=True,
            photometric="minisblack",
            metadata=metadata,
        )
    return buf.getvalue()


def export_presentation_image(
    rgb_uint8: np.ndarray,
    output_format: str = "PNG",
) -> Tuple[bytes, str, str]:
    """
    Encode an 8-bit RGB visual presentation array (`dtype=np.uint8`, shape `(H, W, 3)`)
    with all Lightroom-style tonal adjustments, clipping, and pseudocolors baked in.

    Parameters
    ----------
    rgb_uint8 : np.ndarray
        3-channel 8-bit RGB array (`np.uint8`).
    output_format : str
        One of `'PNG'`, `'JPEG'`, or `'TIFF (8-bit RGB)'`.

    Returns
    -------
    Tuple[bytes, str, str]
        `(file_bytes, mime_type, file_extension)`
    """
    render_arr = np.asarray(rgb_uint8, dtype=np.uint8)
    fmt_upper = output_format.strip().upper()
    buf = io.BytesIO()

    if fmt_upper == "JPEG" or fmt_upper == "JPG":
        pil_img = Image.fromarray(render_arr)
        pil_img.save(buf, format="JPEG", quality=95, subsampling=0)
        return buf.getvalue(), "image/jpeg", "jpg"
    elif "TIFF" in fmt_upper or "TIF" in fmt_upper:
        tifffile.imwrite(buf, render_arr, photometric="rgb")
        return buf.getvalue(), "image/tiff", "tif"
    else:
        pil_img = Image.fromarray(render_arr)
        pil_img.save(buf, format="PNG")
        return buf.getvalue(), "image/png", "png"

