"""CPU regression tests; run with python -m unittest discover -s tests."""

import gzip
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from r2_gaussian.utils.volume_export_utils import save_center_slices, save_nifti_streaming

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


class VolumeOutputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shape = (7, 9, 5)  # Asymmetry exposes XYZ/ZYX swaps.
        x, y, z = np.indices(self.shape)
        self.values = (100 * x + 10 * y + z).astype(np.float32)
        self.spacing = (0.3, 0.7, 1.1)
        self.origin = (-2.1, 4.2, -6.3)

    def mapped_volume(self, dtype):
        path = self.root / f"input_{dtype}.npy"
        np.save(path, self.values.astype(dtype))
        return np.load(path, mmap_mode="r")

    def test_nifti_header_and_streamed_data(self):
        for dtype in ("float32", "float16"):
            with self.subTest(dtype=dtype):
                volume = self.mapped_volume(dtype)
                path = self.root / f"output_{dtype}.nii.gz"
                save_nifti_streaming(volume, self.spacing, self.origin, path)
                with gzip.open(path, "rb") as handle:
                    payload = handle.read()
                nifti_shape = (self.shape[1], self.shape[0], self.shape[2])
                self.assertEqual(struct.unpack_from("<i", payload)[0], 348)
                self.assertEqual(struct.unpack_from("<8h", payload, 40), (3, *nifti_shape, 1, 1, 1, 1))
                self.assertEqual(struct.unpack_from("<2h", payload, 70), (16, 32))
                self.assertEqual(payload[344:352], b"n+1\x00\x00\x00\x00\x00")
                actual = np.frombuffer(payload, dtype="<f4", offset=352).reshape(nifti_shape, order="F")
                np.testing.assert_array_equal(actual, volume.transpose(1, 0, 2))
                del volume

    @unittest.skipIf(sitk is None, "SimpleITK is not installed")
    def test_slicer_reader_geometry_and_orientation(self):
        volume = self.mapped_volume("float16")
        path = self.root / "slicer.nii.gz"
        save_nifti_streaming(volume, self.spacing, self.origin, path)
        image = sitk.ReadImage(str(path))
        self.assertEqual(image.GetSize(), (self.shape[1], self.shape[0], self.shape[2]))
        np.testing.assert_allclose(image.GetSpacing(), self.spacing, rtol=1e-6)
        np.testing.assert_allclose(image.GetOrigin(), self.origin, rtol=1e-6)
        np.testing.assert_allclose(image.GetDirection(), np.eye(3).ravel(), atol=1e-6)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image), volume.transpose(2, 0, 1))
        for index in ((0, 0, 0), (8, 6, 4), (4, 3, 2)):
            expected = np.asarray(self.origin) + np.asarray(index) * self.spacing
            np.testing.assert_allclose(image.TransformIndexToPhysicalPoint(index), expected, rtol=1e-6)
            i, j, k = index
            self.assertEqual(image.GetPixel(index), float(volume[j, i, k]))
        del volume

    @unittest.skipIf(sitk is None, "SimpleITK is not installed")
    def test_matches_original_test_py_nifti_writer(self):
        # A non-cubic asymmetric phantom catches an XY swap even when the
        # original workflow normally uses cubic volumes and isotropic voxels.
        for dtype in ("float32", "float16"):
            with self.subTest(dtype=dtype):
                volume = self.mapped_volume(dtype)
                original = sitk.GetImageFromArray(
                    np.asarray(volume, dtype=np.float32).transpose(2, 0, 1)
                )
                original.SetSpacing(self.spacing)
                original.SetOrigin(self.origin)
                original_path = self.root / f"original_{dtype}.nii.gz"
                sitk.WriteImage(original, str(original_path))
                original_readback = sitk.ReadImage(str(original_path))
                output_path = self.root / f"compat_{dtype}.nii.gz"
                save_nifti_streaming(volume, self.spacing, self.origin, output_path)
                readback = sitk.ReadImage(str(output_path))
                self.assertEqual(readback.GetSize(), original_readback.GetSize())
                np.testing.assert_allclose(readback.GetSpacing(), original_readback.GetSpacing())
                np.testing.assert_allclose(readback.GetOrigin(), original_readback.GetOrigin())
                np.testing.assert_allclose(readback.GetDirection(), original_readback.GetDirection())
                np.testing.assert_array_equal(
                    sitk.GetArrayFromImage(readback), sitk.GetArrayFromImage(original_readback)
                )
                del volume

    def test_center_montage_and_constant_window(self):
        volume = self.mapped_volume("float32")
        path = self.root / "centers.png"
        info = save_center_slices(volume, self.spacing, path)
        self.assertEqual(info["center_indices_xyz"], [3, 4, 2])
        self.assertLess(*info["display_window"])
        self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        from matplotlib import image as mpimg

        preview = mpimg.imread(path)
        self.assertGreater(preview.shape[1], preview.shape[0])
        constant_info = save_center_slices(np.ones(self.shape), self.spacing, self.root / "constant.png")
        self.assertLess(*constant_info["display_window"])
        del volume

    def test_existing_outputs_are_preserved(self):
        for suffix in (".png", ".nii.gz"):
            path = self.root / ("existing" + suffix)
            path.write_bytes(b"preserve me")
            with self.assertRaises(FileExistsError):
                if suffix == ".png":
                    save_center_slices(self.values, self.spacing, path)
                else:
                    save_nifti_streaming(self.values, self.spacing, self.origin, path)
            self.assertEqual(path.read_bytes(), b"preserve me")


if __name__ == "__main__":
    unittest.main()
