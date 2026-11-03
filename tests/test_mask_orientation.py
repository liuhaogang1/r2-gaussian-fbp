"""Physical and array roundtrips for new XYZ and legacy YXZ labelmaps."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from apply_mask_to_volume import load_slicer_mask
from r2_gaussian.utils.volume_export_utils import save_nifti_xyz, volume_geometry

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(sitk is None, "SimpleITK required")
class MaskOrientationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.shape = (7, 9, 11)
        self.mask = np.zeros(self.shape, np.uint16)
        self.mask[1, 6, 4] = 300
        self.mask[5, 2, 7] = 1
        self.scanner = {"nVoxel": list(self.shape), "sVoxel": [1.4, 2.7, 4.4],
                        "offOrigin": [.4, -.6, .9]}

    def write_legacy(self, values, path):
        image = sitk.GetImageFromArray(values.transpose(2, 0, 1))
        image.SetSpacing((.2, .3, .4))
        sitk.WriteImage(image, str(path))

    def test_new_xyz_marker_and_large_labels_survive_resampling(self):
        reference = self.root / "reference.nii.gz"
        labels = self.root / "labels.nii.gz"
        save_nifti_xyz(self.mask, self.scanner, reference)
        image = sitk.ReadImage(str(reference))
        image = sitk.Cast(image, sitk.sitkUInt16)
        sitk.WriteImage(image, str(labels))
        for ref in (None, reference):
            actual, _ = load_slicer_mask(labels, ref, volume_shape=self.shape)
            np.testing.assert_array_equal(actual, self.mask > 0)

    def test_legacy_full_grid_can_be_inferred_and_explicitly_selected(self):
        path = self.root / "legacy.nii.gz"
        self.write_legacy(self.mask, path)
        for axes in ("auto", "legacy_yxz"):
            actual, _ = load_slicer_mask(path, None, axes, self.shape)
            np.testing.assert_array_equal(actual, self.mask > 0)

    def test_cubic_unmarked_reference_requires_explicit_axes(self):
        cube = np.zeros((9, 9, 9), np.uint16)
        cube[1, 6, 4] = 300
        path = self.root / "cube.nii.gz"
        self.write_legacy(cube, path)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            load_slicer_mask(path, None, volume_shape=cube.shape)
        actual, _ = load_slicer_mask(path, None, "legacy_yxz", cube.shape)
        np.testing.assert_array_equal(actual, cube > 0)
        save_nifti_xyz(cube, self.scanner, self.root / "cube_xyz.nii.gz")
        actual, _ = load_slicer_mask(self.root / "cube_xyz.nii.gz", None, volume_shape=cube.shape)
        np.testing.assert_array_equal(actual, cube > 0)

    def test_cropped_and_reoriented_slicer_labelmaps_use_reference_geometry(self):
        reference = self.root / "reference.nii.gz"
        save_nifti_xyz(self.mask, self.scanner, reference)
        image = sitk.ReadImage(str(reference))
        crop = sitk.RegionOfInterest(image, [5, 6, 6], [1, 1, 3])
        oriented = sitk.Flip(sitk.PermuteAxes(image, [1, 0, 2]), [True, False, True])
        for name, modified in (("crop", crop), ("oriented", oriented)):
            path = self.root / (name + ".nii.gz")
            sitk.WriteImage(modified, str(path))
            actual, _ = load_slicer_mask(path, reference, volume_shape=self.shape)
            np.testing.assert_array_equal(actual, self.mask > 0)

    def test_wrong_reference_shape_rejected_before_resampling(self):
        path = self.root / "small.nii.gz"
        save_nifti_xyz(self.mask[:3], self.scanner, path)
        with self.assertRaisesRegex(ValueError, "Reference size"):
            load_slicer_mask(path, None, volume_shape=self.shape)

    def run_cli(self, source, mask, output, extra=()):
        result = subprocess.run([sys.executable, str(ROOT / "apply_mask_to_volume.py"),
            "--volume", str(source), "--mask", str(mask), "--output", str(output),
            "--nii_output", str(output.with_suffix(".nii.gz")), *extra],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cli_new_and_legacy_masks_produce_same_xyz_and_dataset_geometry(self):
        source = self.root / "vol_fbp.npy"
        volume = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
        np.save(source, volume)
        (self.root / "meta_data.json").write_text(json.dumps({"scanner": self.scanner}))
        for name in ("xyz", "legacy"):
            mask_path = self.root / (name + ".nii.gz")
            if name == "xyz":
                save_nifti_xyz(self.mask, self.scanner, mask_path)
            else:
                self.write_legacy(self.mask, mask_path)
            output = self.root / (name + "_masked.npy")
            self.run_cli(source, mask_path, output)
            expected = np.where(self.mask > 0, volume, 0)
            np.testing.assert_array_equal(np.load(output), expected)
            image = sitk.ReadImage(str(output.with_suffix(".nii.gz")))
            np.testing.assert_array_equal(sitk.GetArrayFromImage(image).transpose(2, 1, 0), expected)
            spacing, origin = volume_geometry(self.shape, self.scanner)
            np.testing.assert_allclose(image.GetSpacing(), spacing)
            np.testing.assert_allclose(image.GetOrigin(), origin, atol=1e-6)
            np.testing.assert_allclose(image.GetDirection(), np.eye(3).ravel())

    def test_output_without_npy_suffix_cannot_overwrite_input(self):
        source = self.root / 'source.npy'
        np.save(source, np.ones(self.shape, np.float32))
        before = source.read_bytes()
        mask = self.root / 'mask.nii.gz'
        save_nifti_xyz(self.mask, self.scanner, mask)
        result = subprocess.run([sys.executable, str(ROOT / 'apply_mask_to_volume.py'),
            '--volume', str(source), '--mask', str(mask), '--output', str(source.with_suffix(''))],
            cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must not overwrite', result.stderr)
        self.assertEqual(source.read_bytes(), before)

    def test_no_metadata_preserves_legacy_physical_mapping_with_xyz_payload(self):
        source = self.root / "volume.npy"
        np.save(source, np.ones(self.shape, np.float32))
        mask_path = self.root / "legacy.nii.gz"
        self.write_legacy(self.mask, mask_path)
        output = self.root / "masked.npy"
        self.run_cli(source, mask_path, output)
        image = sitk.ReadImage(str(output.with_suffix(".nii.gz")))
        original = sitk.ReadImage(str(mask_path))
        for xyz in ((1, 6, 4), (5, 2, 7)):
            np.testing.assert_allclose(image.TransformIndexToPhysicalPoint(xyz),
                                       original.TransformIndexToPhysicalPoint((xyz[1], xyz[0], xyz[2])))
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image).transpose(2, 1, 0), self.mask > 0)


if __name__ == "__main__":
    unittest.main()
