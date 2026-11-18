"""Run parallel forward-projection regression checks from the project root."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.forward_project_volume import load_scanner, run, sampling_angles


class ParallelProjectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.volume = self.root / "vol_fbp_raw.npy"
        self.meta = self.root / "meta_data.json"
        self.scanner = dict(mode="parallel", nVoxel=[21, 25, 17], sVoxel=[2.1, 2.5, 1.7],
                            offOrigin=[0, 0, 0], nDetector=[31, 39], sDetector=[3.1, 3.9],
                            offDetector=[0, 0], DSO=5., DSD=8., accuracy=.5)
        self.meta.write_text(json.dumps({"scanner": self.scanner}))

    def test_angle_grid(self):
        for step, count, last in [("0.5", 361, 180), ("0.1", 1801, 180),
                                  ("7", 26, 175), ("200", 1, 0)]:
            grid = sampling_angles(step)
            self.assertEqual((len(grid), grid[0], grid[-1]), (count, 0, last))
            np.testing.assert_allclose(np.diff(grid), float(step))
        for step in ("0", "-1", "NaN", "Infinity", "abc"):
            with self.assertRaises(ValueError):
                sampling_angles(step)

    def test_reject_mismatched_grid_and_cone(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            load_scanner(self.volume, (2, 3, 4), None)
        self.scanner["mode"] = "cone"
        self.meta.write_text(json.dumps({"scanner": self.scanner}))
        with self.assertRaisesRegex(ValueError, "parallel"):
            load_scanner(self.volume, (21, 25, 17), None)

    def test_real_projector(self):
        try:
            from tigre.utilities.gpu import getGpuNames
        except ImportError:
            self.skipTest("TIGRE unavailable")
        if not getGpuNames():
            self.skipTest("CUDA GPU unavailable")
        import tifffile

        values = np.full((21, 25, 17), -1, np.float32)
        # The positive block is centered at physical (+0.4, -0.5, +0.3).
        values[12:17, 5:10, 9:14] = 12
        np.save(self.volume, values)
        args = argparse.Namespace(volume=self.volume, meta=None, step_deg="90",
                                  out=self.root / "base", tiff=True)
        run(args)
        result = np.load(args.out / "projections.npy")
        np.testing.assert_array_equal(tifffile.imread(args.out / "projections.tif"), result)
        np.testing.assert_array_equal(np.load(args.out / "angles_deg.npy"), [0, 90, 180])
        np.testing.assert_allclose(result[:, 15, 19], [-2.1, -2.5, -2.1], atol=.05)
        for image, column in zip(result, [14, 15, 24]):
            row_col = np.argwhere(image > 0).mean(axis=0)
            np.testing.assert_allclose(row_col, [12, column], atol=.6)
        with self.assertRaises(FileExistsError):
            run(args)

        # Export with the same physical bounds and twice the XYZ sampling.
        export = self.root / "highres"
        export.mkdir()
        args.volume = export / "vol_pred.npy"
        dense = values.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)
        np.save(args.volume, dense)
        args.volume.with_suffix(".json").write_text(json.dumps({
            "volume": args.volume.name, "shape_xyz": dense.shape, "source_path": str(self.root),
            "nifti_geometry": {"shape_xyz": dense.shape, "size_xyz": self.scanner["sVoxel"],
                               "center_xyz": self.scanner["offOrigin"]},
        }))
        args.out = self.root / "dense"
        run(args)
        np.testing.assert_allclose(np.load(args.out / "projections.npy"), result, atol=.05)


if __name__ == "__main__":
    unittest.main()
