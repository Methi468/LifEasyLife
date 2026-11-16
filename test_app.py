"""
Unit tests for the Leica .lif Microscopy Viewer & Real-Time Studio.
Verifies strict isolation between 'Save for Analysis' (Raw Data via tifffile)
and 'Save for Presentation' (8-bit RGB Visual Render: PNG, JPEG, TIFF).
"""

from collections import namedtuple
import io
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import tifffile

from image_processing import (
    FLUOROPHORE_PRESETS,
    ChannelSettings,
    apply_lightroom_adjustments,
    blend_channels,
    compute_z_projection,
    draw_scale_bar,
    export_presentation_image,
    export_raw_analysis_tiff,
    extract_raw_analysis_plane,
    hex_to_rgb_norm,
)
import lif_loader
from realtime_studio import build_realtime_studio_html


DimsTuple = namedtuple("Dims", ["x", "y", "z", "t", "m"])


class TestImageProcessingAndExportIsolation(unittest.TestCase):
    def test_alexa_presets_present(self):
        self.assertIn("Alexa 568 (Orange-Red)", FLUOROPHORE_PRESETS)
        self.assertIn("Alexa 647 (Magenta)", FLUOROPHORE_PRESETS)
        self.assertIn("Alexa 647 (Far-Red)", FLUOROPHORE_PRESETS)
        self.assertIn("Alexa 488 / GFP (Green)", FLUOROPHORE_PRESETS)

        r, g, b = hex_to_rgb_norm(FLUOROPHORE_PRESETS["Alexa 568 (Orange-Red)"].hex_color)
        self.assertAlmostEqual(r, 1.0)
        self.assertGreater(g, 0.2)
        self.assertEqual(b, 0.0)

    def test_maximum_intensity_projection(self):
        stack = np.zeros((5, 4, 4), dtype=np.uint16)
        stack[0, 0, 0] = 10
        stack[2, 0, 0] = 3950
        stack[4, 1, 2] = 1042

        mip = compute_z_projection(stack, use_mip=True)
        self.assertEqual(mip.shape, (4, 4))
        self.assertEqual(mip[0, 0], 3950.0)
        self.assertEqual(mip[1, 2], 1042.0)

        slice_0 = compute_z_projection(stack, use_mip=False, z_index=0)
        self.assertEqual(slice_0[0, 0], 10.0)
        self.assertEqual(slice_0[1, 2], 0.0)

    def test_strict_isolation_between_raw_analysis_and_presentation(self):
        # Create 16-bit raw microscopy data (e.g. 12-bit values in uint16 container)
        raw_ch0 = np.array([[[100, 1200], [2500, 4095]]], dtype=np.uint16)
        raw_ch1 = np.array([[[50, 600], [1800, 3500]]], dtype=np.uint16)
        raw_ch0.flags.writeable = False
        raw_ch1.flags.writeable = False

        # Extract 2D raw analysis planes
        raw_plane0 = extract_raw_analysis_plane(raw_ch0, use_mip=True)
        raw_plane1 = extract_raw_analysis_plane(raw_ch1, use_mip=True)
        self.assertEqual(raw_plane0.dtype, np.uint16)
        self.assertFalse(raw_plane0.flags.writeable)

        # Apply extreme Lightroom adjustments for presentation
        vis_plane0 = compute_z_projection(raw_ch0, use_mip=True)
        vis_plane1 = compute_z_projection(raw_ch1, use_mip=True)
        cfg0 = ChannelSettings(
            enabled=True,
            custom_hex="#FF4500",
            brightness=65.0,
            contrast=70.0,
            exposure=1.5,
            gamma=1.8,
            black_point=200.0,
            white_point=3000.0,
        )
        cfg1 = ChannelSettings(
            enabled=True,
            custom_hex="#FF00FF",
            brightness=-30.0,
            contrast=40.0,
            black_point=100.0,
            white_point=2500.0,
        )
        adj0 = apply_lightroom_adjustments(vis_plane0, cfg0)
        adj1 = apply_lightroom_adjustments(vis_plane1, cfg1)
        pres_rgb8 = blend_channels([adj0, adj1], [cfg0, cfg1])
        self.assertEqual(pres_rgb8.dtype, np.uint8)
        self.assertEqual(pres_rgb8.shape, (2, 2, 3))

        # 1. Verify 'Save for Analysis' (Raw Data via tifffile, imagej=True)
        # preserves exact uint16 values as independent (C, Y, X) ImageJ hyperstack layers
        raw_tiff_bytes = export_raw_analysis_tiff(
            [raw_plane0, raw_plane1],
            selected_channel_indices=[0, 1],
            channel_names=["Ch1", "Ch2"],
            scale_x=5.58,
            scale_y=5.58,
        )
        with tifffile.TiffFile(io.BytesIO(raw_tiff_bytes)) as tf:
            self.assertTrue(tf.is_imagej)
            self.assertTrue(tf.imagej_metadata.get("hyperstack"))
            self.assertEqual(tf.imagej_metadata.get("channels"), 2)
            recovered_raw = tf.asarray()

        self.assertEqual(recovered_raw.dtype, np.uint16)
        self.assertEqual(recovered_raw.shape, (2, 2, 2))  # (C, Y, X)
        np.testing.assert_array_equal(recovered_raw[0], raw_ch0[0])
        np.testing.assert_array_equal(recovered_raw[1], raw_ch1[0])

        # Also verify 3D Z-stack raw analysis export produces (Z, C, Y, X) ImageJ hyperstack
        raw_z0 = np.stack([raw_ch0[0], raw_ch0[0] + 10, raw_ch0[0] + 20], axis=0)
        raw_z1 = np.stack([raw_ch1[0], raw_ch1[0] + 30, raw_ch1[0] + 40], axis=0)
        raw_3d_bytes = export_raw_analysis_tiff(
            [raw_z0, raw_z1],
            selected_channel_indices=[0, 1],
            channel_names=["Ch1", "Ch2"],
            scale_x=5.58,
            scale_y=5.58,
            scale_z=2.0,
        )
        with tifffile.TiffFile(io.BytesIO(raw_3d_bytes)) as tf3d:
            self.assertTrue(tf3d.is_imagej)
            self.assertTrue(tf3d.imagej_metadata.get("hyperstack"))
            self.assertEqual(tf3d.imagej_metadata.get("channels"), 2)
            self.assertEqual(tf3d.imagej_metadata.get("slices"), 3)
            recovered_3d = tf3d.asarray()

        self.assertEqual(recovered_3d.shape, (3, 2, 2, 2))  # (Z, C, Y, X)
        np.testing.assert_array_equal(recovered_3d[:, 0], raw_z0)
        np.testing.assert_array_equal(recovered_3d[:, 1], raw_z1)

        # 2. Verify 'Save for Presentation' supports PNG, JPEG, and TIFF (8-bit RGB)
        png_bytes, png_mime, png_ext = export_presentation_image(pres_rgb8, "PNG")
        self.assertEqual(png_mime, "image/png")
        self.assertEqual(png_ext, "png")
        recovered_png = np.asarray(Image.open(io.BytesIO(png_bytes)))
        np.testing.assert_array_equal(recovered_png, pres_rgb8)

        jpg_bytes, jpg_mime, jpg_ext = export_presentation_image(pres_rgb8, "JPEG")
        self.assertEqual(jpg_mime, "image/jpeg")
        self.assertEqual(jpg_ext, "jpg")
        recovered_jpg = np.asarray(Image.open(io.BytesIO(jpg_bytes)))
        self.assertEqual(recovered_jpg.shape, (2, 2, 3))
        self.assertEqual(recovered_jpg.dtype, np.uint8)

        tif8_bytes, tif8_mime, tif8_ext = export_presentation_image(
            pres_rgb8, "TIFF (8-bit RGB)"
        )
        self.assertEqual(tif8_mime, "image/tiff")
        self.assertEqual(tif8_ext, "tif")
        recovered_tif8 = tifffile.imread(io.BytesIO(tif8_bytes))
        self.assertEqual(recovered_tif8.dtype, np.uint8)
        np.testing.assert_array_equal(recovered_tif8, pres_rgb8)

    def test_blend_channels_and_scale_bar(self):
        p1 = np.ones((64, 64), dtype=np.float32) * 0.5
        p2 = np.ones((64, 64), dtype=np.float32) * 0.8
        c1 = ChannelSettings(enabled=True, custom_hex="#FF0000")
        c2 = ChannelSettings(enabled=True, custom_hex="#00FF00")

        comp = blend_channels([p1, p2], [c1, c2])
        self.assertEqual(comp.shape, (64, 64, 3))
        self.assertEqual(comp.dtype, np.uint8)
        self.assertAlmostEqual(int(comp[0, 0, 0]), 128, delta=1)
        self.assertAlmostEqual(int(comp[0, 0, 1]), 204, delta=1)
        self.assertEqual(int(comp[0, 0, 2]), 0)

        with_bar = draw_scale_bar(comp, scale_px_per_um=2.0)
        self.assertEqual(with_bar.shape, (64, 64, 3))


class TestLifLoader(unittest.TestCase):
    def test_parse_and_load_with_mock_readlif_preserves_uint16(self):
        mock_img = MagicMock()
        mock_img.name = "Sample_ZStack"
        mock_img.path = "Project/Sample_ZStack"
        mock_img.dims = DimsTuple(x=16, y=12, z=3, t=1, m=1)
        mock_img.channels = 2
        mock_img.bit_depth = (12, 12)
        mock_img.scale = (5.58, 5.58, 1.5, None)
        mock_img.settings = {"Magnification": "40", "ObjectiveName": "ACS APO 40x/1.15 OIL", "Zoom": "1.5"}

        def fake_get_frame(z=0, t=0, c=0, m=0):
            arr = np.full((12, 16), fill_value=(c + 1) * 1000 + z, dtype=np.uint16)
            return Image.fromarray(arr)

        mock_img.get_frame.side_effect = fake_get_frame

        mock_lif_instance = MagicMock()
        mock_lif_instance.num_images = 1
        mock_lif_instance.xml_header = (
            "<LMSDataContainerHeader><Element Name='Sample_ZStack'>"
            "<Data><Image><ImageDescription><Channels>"
            "<ChannelDescription LUTName='Alexa 568'/>"
            "<ChannelDescription LUTName='Alexa 647'/>"
            "</Channels></ImageDescription></Image></Data>"
            "</Element></LMSDataContainerHeader>"
        )
        mock_lif_instance.get_image.return_value = mock_img

        with patch.object(lif_loader, "READLIF_AVAILABLE", True), patch.object(
            lif_loader, "LifFile", return_value=mock_lif_instance
        ):
            series = lif_loader.parse_lif_series_list("dummy.lif")
            self.assertEqual(len(series), 1)
            self.assertEqual(series[0].name, "Sample_ZStack")
            self.assertEqual(series[0].dim_z, 3)
            self.assertEqual(series[0].num_channels, 2)
            self.assertEqual(series[0].objective_magnification, 40.0)

            stacks = lif_loader.load_series_channel_stacks("dummy.lif", 0)
            self.assertEqual(len(stacks), 2)
            self.assertEqual(stacks[0].shape, (3, 12, 16))
            self.assertEqual(stacks[0].dtype, np.uint16)
            self.assertFalse(stacks[0].flags.writeable)
            self.assertEqual(int(stacks[0][2, 0, 0]), 1002)
            self.assertEqual(int(stacks[1][2, 0, 0]), 2002)

    def test_synthetic_demo_series_and_studio_html(self):
        meta_list, stacks_map = lif_loader.generate_synthetic_demo_series()
        self.assertGreaterEqual(len(meta_list), 2)
        self.assertTrue(meta_list[0].is_z_stack)
        self.assertEqual(stacks_map[0][0].dtype, np.uint16)
        self.assertFalse(stacks_map[0][0].flags.writeable)

        vis_planes = [compute_z_projection(s, use_mip=True) for s in stacks_map[0]]
        raw_planes = [extract_raw_analysis_plane(s, use_mip=True) for s in stacks_map[0]]
        html = build_realtime_studio_html(
            "demo", meta_list[0], vis_planes, "MIP", raw_analysis_planes_2d=raw_planes
        )
        self.assertIn("Save for Analysis (Raw Data)", html)
        self.assertIn("Save for Presentation (Visual Render)", html)
        self.assertIn("pres-format-select", html)
        self.assertIn("downloadRawAnalysisTIFF", html)
        self.assertIn("downloadPresentationImage", html)


if __name__ == "__main__":
    unittest.main()
