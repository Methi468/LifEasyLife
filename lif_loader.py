"""
Leica .lif file parser and Z-stack extractor powered by `readlif`.

Uses `readlif.reader.LifFile` to inspect series, dimensions (X, Y, Z, T, M),
channels, bit depths, physical scales, objective magnification, and scanner zoom,
and to extract 3D (Z, Y, X) arrays for each channel.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import numpy as np

try:
    from readlif.reader import LifFile
    READLIF_AVAILABLE = True
except ImportError:
    LifFile = None  # type: ignore
    READLIF_AVAILABLE = False


@dataclass
class SeriesMetadata:
    """Metadata for a single image series inside a Leica .lif file."""
    index: int
    name: str
    path: str
    dim_x: int
    dim_y: int
    dim_z: int
    dim_t: int
    dim_m: int
    num_channels: int
    bit_depths: Tuple[int, ...]
    scale_x: Optional[float] = None  # pixels per micrometer (px/µm)
    scale_y: Optional[float] = None
    scale_z: Optional[float] = None
    objective_magnification: Optional[float] = None  # e.g., 10.0, 40.0, 60.0, 63.0
    objective_name: str = ""                         # e.g., 'ACS APO 40x/1.15 OIL'
    scanner_zoom: float = 1.0                        # e.g., 1.5
    channel_names: List[str] = field(default_factory=list)
    channel_luts: List[str] = field(default_factory=list)

    @property
    def is_z_stack(self) -> bool:
        """Return True if this series contains multiple Z-slices."""
        return self.dim_z > 1

    @property
    def display_label(self) -> str:
        """Human-readable summary label for series selection dropdowns."""
        z_label = f"{self.dim_z} Z-slices" if self.is_z_stack else "Single Z-plane"
        t_label = f", {self.dim_t}T" if self.dim_t > 1 else ""
        m_label = f", {self.dim_m} tiles" if self.dim_m > 1 else ""
        obj_label = (
            f", {int(self.objective_magnification)}x"
            if self.objective_magnification
            else ""
        )
        return (
            f"Series {self.index + 1}: {self.name} "
            f"({self.dim_x}×{self.dim_y} px, {self.num_channels} ch, {z_label}{obj_label}{t_label}{m_label})"
        )


def _extract_xml_channel_luts(xml_header: Optional[str]) -> Dict[str, List[str]]:
    """
    Attempt to parse LUTName attributes per image Element from the Leica XML header.
    Returns a mapping from element name to a list of LUT names per channel.
    """
    if not xml_header:
        return {}
    result: Dict[str, List[str]] = {}
    try:
        root = ET.fromstring(xml_header)
        for elem in root.iter("Element"):
            elem_name = elem.attrib.get("Name")
            if not elem_name:
                continue
            luts: List[str] = []
            for ch_desc in elem.findall(".//ImageDescription/Channels/ChannelDescription"):
                lut_name = ch_desc.attrib.get("LUTName", "")
                luts.append(lut_name)
            if luts:
                result[elem_name] = luts
    except Exception:
        pass
    return result


def map_lut_hint_to_preset(lut_name: str, channel_idx: int) -> str:
    """Map a Leica LUTName string (e.g. 'Green', 'Red', 'Blue', 'Cyan', 'Magenta') to a preset."""
    lower = (lut_name or "").strip().lower()
    if "blue" in lower or "dapi" in lower or "405" in lower:
        return "Alexa 405 / DAPI (Blue)"
    if "green" in lower or "488" in lower or "fitc" in lower or "gfp" in lower:
        return "Alexa 488 / GFP (Green)"
    if "568" in lower or "orange" in lower:
        return "Alexa 568 (Orange-Red)"
    if "546" in lower or "cy3" in lower or "tritc" in lower:
        return "Alexa 546 / Cy3 (Yellow-Orange)"
    if "594" in lower or "red" in lower or "texas" in lower or "mcherry" in lower:
        return "Alexa 568 (Orange-Red)" if channel_idx == 1 else "Alexa 594 / mCherry (Red)"
    if "647" in lower or "magenta" in lower or "cy5" in lower or "far" in lower:
        return "Alexa 647 (Magenta)"
    if "cyan" in lower or "aqua" in lower or "cfp" in lower:
        return "Cyan (CFP / Aqua)"
    if "yellow" in lower or "yfp" in lower:
        return "Yellow (YFP / Gold)"
    if "gray" in lower or "grey" in lower or "white" in lower:
        return "Grayscale / White"

    fallback_order = [
        "Alexa 488 / GFP (Green)",
        "Alexa 568 (Orange-Red)",
        "Alexa 647 (Magenta)",
        "Alexa 405 / DAPI (Blue)",
        "Cyan (CFP / Aqua)",
        "Yellow (YFP / Gold)",
    ]
    return fallback_order[channel_idx % len(fallback_order)]


def parse_lif_series_list(file_path: str) -> List[SeriesMetadata]:
    """
    Parse a Leica .lif file using `readlif.reader.LifFile` and return metadata for all series.
    """
    if not READLIF_AVAILABLE or LifFile is None:
        raise ImportError(
            "The `readlif` library is required to parse .lif files. "
            "Install it via `pip install readlif`."
        )

    lif = LifFile(file_path)
    xml_luts_by_name = _extract_xml_channel_luts(getattr(lif, "xml_header", None))
    series_list: List[SeriesMetadata] = []

    for idx in range(lif.num_images):
        img = lif.get_image(idx)
        dims = img.dims
        dim_x = int(getattr(dims, "x", dims[0]) or 1)
        dim_y = int(getattr(dims, "y", dims[1]) or 1)
        dim_z = int(getattr(dims, "z", dims[2]) or 1)
        dim_t = int(getattr(dims, "t", dims[3]) or 1)
        dim_m = int(getattr(dims, "m", dims[4]) if len(dims) > 4 else 1) or 1

        num_channels = int(getattr(img, "channels", 1) or 1)
        raw_bit_depth = getattr(img, "bit_depth", None)
        if isinstance(raw_bit_depth, (tuple, list)) and len(raw_bit_depth) > 0:
            bit_depths = tuple(int(b) for b in raw_bit_depth)
        elif isinstance(raw_bit_depth, int):
            bit_depths = tuple([raw_bit_depth] * num_channels)
        else:
            bit_depths = tuple([8] * num_channels)

        scale = getattr(img, "scale", (None, None, None, None))
        scale_x = abs(float(scale[0])) if len(scale) > 0 and scale[0] else None
        scale_y = abs(float(scale[1])) if len(scale) > 1 and scale[1] else None
        scale_z = abs(float(scale[2])) if len(scale) > 2 and scale[2] else None

        # Extract Leica microscope hardware settings (Magnification, ObjectiveName, Zoom)
        settings = getattr(img, "settings", {}) or {}
        obj_mag: Optional[float] = None
        if isinstance(settings, dict):
            raw_mag = settings.get("Magnification")
            if raw_mag is not None:
                try:
                    obj_mag = float(raw_mag)
                except (ValueError, TypeError):
                    obj_mag = None
            obj_name = str(settings.get("ObjectiveName", "")).strip()
            raw_zoom = settings.get("Zoom")
            try:
                scanner_zoom = float(raw_zoom) if raw_zoom is not None else 1.0
            except (ValueError, TypeError):
                scanner_zoom = 1.0
        else:
            obj_name = ""
            scanner_zoom = 1.0

        name = str(getattr(img, "name", f"Series_{idx + 1}"))
        short_name = name.split("/")[-1]
        path_val = str(getattr(img, "path", name))

        luts = xml_luts_by_name.get(name) or xml_luts_by_name.get(short_name) or []
        ch_names = []
        ch_luts = []
        for c in range(num_channels):
            lut_hint = luts[c] if c < len(luts) else ""
            ch_luts.append(lut_hint)
            if lut_hint:
                ch_names.append(f"Channel {c + 1} ({lut_hint})")
            else:
                ch_names.append(f"Channel {c + 1}")

        series_list.append(
            SeriesMetadata(
                index=idx,
                name=name,
                path=path_val,
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                dim_t=dim_t,
                dim_m=dim_m,
                num_channels=num_channels,
                bit_depths=bit_depths,
                scale_x=scale_x,
                scale_y=scale_y,
                scale_z=scale_z,
                objective_magnification=obj_mag,
                objective_name=obj_name,
                scanner_zoom=scanner_zoom,
                channel_names=ch_names,
                channel_luts=ch_luts,
            )
        )

    return series_list


def load_series_channel_stacks(
    file_path: str,
    series_index: int,
    t_index: int = 0,
    m_index: int = 0,
) -> List[np.ndarray]:
    """
    Extract all Z-slices for each channel of the specified series in a .lif file.
    Preserves the exact unmodified integer dtype (`uint8` or `uint16`) returned by
    `readlif` and marks the returned raw arrays as read-only (`writeable = False`)
    so downstream visual rendering can never mutate quantitative raw data.
    """
    if not READLIF_AVAILABLE or LifFile is None:
        raise ImportError("The `readlif` library is required to read .lif files.")

    lif = LifFile(file_path)
    img = lif.get_image(series_index)

    dims = img.dims
    dim_z = max(1, int(getattr(dims, "z", dims[2]) or 1))
    num_channels = max(1, int(getattr(img, "channels", 1) or 1))
    raw_bit_depth = getattr(img, "bit_depth", None)

    channel_stacks: List[np.ndarray] = []

    for c in range(num_channels):
        if isinstance(raw_bit_depth, (tuple, list)) and c < len(raw_bit_depth):
            b_depth = int(raw_bit_depth[c])
        elif isinstance(raw_bit_depth, int):
            b_depth = int(raw_bit_depth)
        else:
            b_depth = 8
        target_dtype = np.uint8 if b_depth <= 8 else np.uint16

        z_planes: List[np.ndarray] = []
        for z in range(dim_z):
            try:
                pil_frame = img.get_frame(z=z, t=t_index, c=c, m=m_index)
            except TypeError:
                pil_frame = img.get_frame(z=z, t=t_index, c=c)
            plane = np.array(pil_frame, dtype=target_dtype, copy=True)
            z_planes.append(plane)
        stack_3d = np.stack(z_planes, axis=0)  # shape: (Z, Y, X), dtype: uint8 or uint16
        stack_3d.flags.writeable = False
        channel_stacks.append(stack_3d)

    return channel_stacks


def generate_synthetic_demo_series() -> Tuple[List[SeriesMetadata], Dict[int, List[np.ndarray]]]:
    """
    Create a realistic synthetic multi-series 3-channel confocal microscopy Z-stack
    dataset so users can explore the Color Adjustment sliders, Alexa 568 / Alexa 647
    pseudocoloring, and Maximum Intensity Projection (MIP) even before uploading a .lif file.
    """
    series_meta = [
        SeriesMetadata(
            index=0,
            name="Confocal_Cells_ZStack_40x",
            path="Experiment_1/Confocal_Cells_ZStack_40x",
            dim_x=320,
            dim_y=320,
            dim_z=15,
            dim_t=1,
            dim_m=1,
            num_channels=3,
            bit_depths=(12, 12, 12),
            scale_x=1.745,
            scale_y=1.745,
            scale_z=1.5,
            objective_magnification=40.0,
            objective_name="ACS APO 40x/1.15 OIL",
            scanner_zoom=1.0,
            channel_names=[
                "Channel 1 (Actin Cytoskeleton - 488)",
                "Channel 2 (Mitochondria - Alexa 568)",
                "Channel 3 (Far-Red Puncta / Nuclei - Alexa 647)",
            ],
            channel_luts=["Green", "Alexa 568", "Alexa 647"],
        ),
        SeriesMetadata(
            index=1,
            name="Single_Focal_Plane_Overview_10x",
            path="Experiment_1/Single_Focal_Plane_Overview_10x",
            dim_x=320,
            dim_y=320,
            dim_z=1,
            dim_t=1,
            dim_m=1,
            num_channels=2,
            bit_depths=(8, 8),
            scale_x=0.436,
            scale_y=0.436,
            scale_z=None,
            objective_magnification=10.0,
            objective_name="HC PL FLUOTAR 10x/0.30",
            scanner_zoom=1.0,
            channel_names=[
                "Channel 1 (Alexa 568)",
                "Channel 2 (Alexa 647)",
            ],
            channel_luts=["Alexa 568", "Alexa 647"],
        ),
    ]

    rng = np.random.default_rng(42)
    z_size, height, width = 15, 320, 320
    zz, yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, z_size, dtype=np.float32),
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )

    ch0 = np.zeros((z_size, height, width), dtype=np.float32)
    centers = [(-0.35, -0.25, -0.4, 0.36), (0.30, 0.28, 0.35, 0.40), (-0.15, 0.35, 0.0, 0.28)]
    for cx, cy, cz, radius in centers:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + ((zz - cz) * 0.85) ** 2)
        shell = np.exp(-((dist - radius) ** 2) / (2 * 0.035 ** 2))
        filaments = 0.5 + 0.5 * np.cos(18.0 * (xx - cx) + 14.0 * (yy - cy) + 6.0 * zz)
        ch0 += shell * filaments * 3400.0

    ch1 = np.zeros((z_size, height, width), dtype=np.float32)
    for i in range(18):
        angle = i * np.pi / 9.0
        z_center = -0.75 + (i / 17.0) * 1.5
        cx = 0.42 * np.cos(angle * 1.7)
        cy = 0.42 * np.sin(angle * 1.3)
        tube = np.exp(
            -(
                ((xx - cx) ** 2) / (2 * 0.14 ** 2)
                + ((yy - cy) ** 2) / (2 * 0.045 ** 2)
                + ((zz - z_center) ** 2) / (2 * 0.14 ** 2)
            )
        )
        ch1 += tube * 3600.0

    ch2 = np.zeros((z_size, height, width), dtype=np.float32)
    for cx, cy, cz, r_xy in [
        (-0.35, -0.25, -0.4, 0.16),
        (0.30, 0.28, 0.35, 0.18),
        (-0.15, 0.35, 0.0, 0.13),
    ]:
        nuc = np.exp(
            -(
                ((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r_xy ** 2)
                + ((zz - cz) ** 2) / (2 * 0.20 ** 2)
            )
        )
        ch2 += nuc * 3800.0

    for _ in range(25):
        px = rng.uniform(-0.75, 0.75)
        py = rng.uniform(-0.75, 0.75)
        pz = rng.uniform(-0.85, 0.85)
        punctum = np.exp(
            -(
                ((xx - px) ** 2 + (yy - py) ** 2) / (2 * 0.025 ** 2)
                + ((zz - pz) ** 2) / (2 * 0.09 ** 2)
            )
        )
        ch2 += punctum * 3200.0

    ch0 = np.clip(np.rint(ch0 + rng.normal(80.0, 25.0, ch0.shape)), 0, 4095).astype(np.uint16)
    ch1 = np.clip(np.rint(ch1 + rng.normal(70.0, 22.0, ch1.shape)), 0, 4095).astype(np.uint16)
    ch2 = np.clip(np.rint(ch2 + rng.normal(65.0, 20.0, ch2.shape)), 0, 4095).astype(np.uint16)

    s1_ch0 = np.clip(np.rint(np.max(ch1, axis=0) / 16.0), 0, 255).astype(np.uint8)[np.newaxis, ...]
    s1_ch1 = np.clip(np.rint(np.max(ch2, axis=0) / 16.0), 0, 255).astype(np.uint8)[np.newaxis, ...]

    for arr in (ch0, ch1, ch2, s1_ch0, s1_ch1):
        arr.flags.writeable = False

    stacks_by_series = {
        0: [ch0, ch1, ch2],
        1: [s1_ch0, s1_ch1],
    }
    return series_meta, stacks_by_series
