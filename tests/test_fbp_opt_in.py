"""Compare disabled options with the pre-feature implementation (964783a).

Uses deterministic CPU reconstruction stand-ins, not a GPU equivalence claim.
The pinned reference is read from Git without altering the working tree.
"""
import argparse
import builtins
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import numpy as np
import tifffile

from fbp_test_support import import_prepare
import init_from_fbp as init
import run_pipeline as pipeline

prepare = import_prepare()
ROOT = Path(__file__).resolve().parents[1]
BASELINE = "964783a"
OPTIONAL_MODULES = ("scripts.auto_svoxel", "scripts.gaussian_fbp", "scripts.fbp_geometry")


def reference(filename):
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{filename}"], cwd=ROOT).decode("utf-8")
    module = ModuleType(f"legacy_{Path(filename).stem}")
    module.__file__ = str(ROOT / filename)
    # Expose the original CLI parser for tests without executing the CLI.
    if filename == "prepare_fbp_tiff.py":
        source, parser = source.rsplit('if __name__ == "__main__":', 1)
        source += '\ndef build_parser():' + parser.replace(
            "    main(parser.parse_args())", "    return parser")
    with mock.patch.dict(sys.modules, {"tigre": prepare.tigre, "tigre.algorithms": prepare.algs}):
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


@contextlib.contextmanager
def forbid_optional_imports():
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in OPTIONAL_MODULES:
            raise AssertionError(f"Disabled feature was imported: {name}")
        return original(name, *args, **kwargs)
    with mock.patch("builtins.__import__", side_effect=guarded):
        yield


class OptInRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_pipeline = reference("run_pipeline.py")
        cls.old_prepare = reference("prepare_fbp_tiff.py")
        cls.old_init = reference("init_from_fbp.py")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_original_resolved_arguments_and_stage_commands(self):
        base = ["--input_dir", str(self.root / "input"), "--output_dir",
                str(self.root / "out"), "--model_path", str(self.root / "model")]
        for extra in ([], ["--fbp_sVoxel", "0.01", "0.02", "0.03",
                           "--fbp_offOrigin", "1", "2", "3"],
                      ["--init_source", "train_fbp", "--train_slice_loss_enabled"],
                      ["--fbp_mode", "cone", "--fbp_recon_pixel_subsample", "4"]):
            with self.subTest(extra=extra), forbid_optional_imports():
                args = pipeline.resolve_args(pipeline.build_parser().parse_args(base + extra))
                old = self.old_pipeline.resolve_args(self.old_pipeline.build_parser().parse_args(base + extra))
                self.assertEqual(vars(args), vars(old))
                pipeline.validate_args(args)
                self.old_pipeline.validate_args(old)
                self.assertEqual(pipeline.build_fbp_command(args, None, [0, 0]),
                                 self.old_pipeline.build_fbp_command(old, None, [0, 0]))
                self.assertEqual(pipeline.build_init_command(args, self.root / "init.npy"),
                                 self.old_pipeline.build_init_command(old, self.root / "init.npy"))

    def test_default_dry_run_manifest_matches_original(self):
        manifests = []
        for module in (self.old_pipeline, pipeline):
            args = module.resolve_args(module.build_parser().parse_args([
                "--input_dir", str(self.root / "input"),
                "--output_dir", str(self.root / "out"),
                "--model_path", str(self.root / "model"),
                "--dry_run", "--skip_center", "--stop_after", "fbp"]))
            with forbid_optional_imports(), contextlib.redirect_stdout(io.StringIO()):
                module.main(args)
            manifests.append(json.loads((args.output_dir / "pipeline_manifest.json").read_text()))
        # Execution timestamps are the only expected nondeterminism.
        for manifest in manifests:
            for key in list(manifest):
                if "time" in key:
                    manifest.pop(key)
        self.assertEqual(manifests[0], manifests[1])

    def test_default_prepare_outputs_geometry_and_previews_match_original(self):
        source = self.root / "input"
        source.mkdir()
        for i in range(4):
            tifffile.imwrite(source / f"Tomo_{i}.tiff",
                             np.arange(12 * 20, dtype=np.float32).reshape(12, 20) + i + 1)
        for factor, recon, mode in ((1, None, "parallel"), (2, None, "parallel"),
                                    (1, 4, "parallel"), (1, None, "cone")):
            results = []
            for name, module in (("baseline", self.old_prepare), ("current", prepare)):
                output = self.root / f"{name}_{factor}_{recon}_{mode}"
                command = ["--input_dir", str(source), "--output_dir", str(output),
                           "--input_type", "line_integral", "--shift_v", "0",
                           "--pixel_subsample", str(factor), "--mode", mode,
                           "--n_train", "3", "--n_test", "1",
                           "--nVoxel", "4", "6", "8", "--sVoxel", "0.01", "0.02", "0.03"]
                if recon is not None:
                    command += ["--recon_pixel_subsample", str(recon)]
                args = module.build_parser().parse_args(command)
                calls = []
                def backend(projections, angles, options, geometry):
                    calls.append((projections.copy(), angles.copy(), vars(geometry).copy()))
                    return np.full(options.nVoxel, projections.mean(), dtype=np.float32)
                with forbid_optional_imports(), contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    module, "reconstruct_volume", side_effect=backend
                ), mock.patch.object(module, "save_fbp_preview", return_value=(0, 1)) as preview:
                    module.main(args)
                metadata = json.loads((output / "meta_data.json").read_text())
                arrays = {str(p.relative_to(output)): np.load(p) for p in output.rglob("*.npy")}
                results.append((metadata, arrays, calls, preview.call_args.args[2]))
            old, new = results
            with self.subTest(factor=factor, recon=recon, mode=mode):
                self.assertEqual(old[0], new[0])
                self.assertEqual(old[1].keys(), new[1].keys())
                for key in old[1]:
                    np.testing.assert_array_equal(old[1][key], new[1][key])
                self.assertEqual(len(old[2]), len(new[2]))
                for old_call, new_call in zip(old[2], new[2]):
                    np.testing.assert_array_equal(old_call[0], new_call[0])
                    np.testing.assert_array_equal(old_call[1], new_call[1])
                    self.assertEqual(old_call[2].keys(), new_call[2].keys())
                    for key in old_call[2]:
                        np.testing.assert_equal(old_call[2][key], new_call[2][key])
                np.testing.assert_array_equal(old[3], new[3])

    def test_metadata_cannot_enable_gaussian_cache_or_signature_checks(self):
        cfg = {"scanner": {"nVoxel": [4, 6, 8]},
               "reconstruction": {"coordinate_mode": "gaussian"}}
        (self.root / "meta_data.json").write_text(json.dumps(cfg))
        expected = np.arange(192, dtype=np.float32).reshape(4, 6, 8)
        np.save(self.root / "vol_train_fbp.npy", expected)
        args = argparse.Namespace(dataset=self.root, use_train_fbp=True, volume=None, fbp_output=None)
        with forbid_optional_imports(), mock.patch.object(init, "reconstruct_from_train_views") as reconstruct:
            actual, output = init.load_initial_volume(args)
        reconstruct.assert_not_called()
        self.assertEqual(output.name, "vol_train_fbp.npy")
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(list(self.root.glob("*.geometry.json")))

    def test_metadata_cannot_enable_gaussian_train_reconstruction(self):
        scanner = {"nVoxel": [4, 6, 8], "sVoxel": [1, 2, 3],
                   "sDetector": [8, 10], "mode": "parallel"}
        (self.root / "meta_data.json").write_text(json.dumps({
            "scanner": scanner, "proj_train": [{"file_path": "view.npy", "angle": 0}],
            "reconstruction": {"coordinate_mode": "gaussian"}}))
        np.save(self.root / "view.npy", np.ones((8, 10), dtype=np.float32))
        backend = ModuleType("r2_gaussian.utils.ct_utils")
        backend.get_geometry_tigre = mock.Mock(side_effect=lambda cfg: cfg)
        backend.recon_volume = mock.Mock(return_value=np.ones((4, 6, 8)))
        with forbid_optional_imports(), mock.patch.dict(sys.modules, {backend.__name__: backend}):
            init.reconstruct_from_train_views(self.root)
        self.assertEqual(backend.get_geometry_tigre.call_args.args[0], scanner)

    def test_default_point_positions_match_original_with_offset_non_cube_metadata(self):
        volume_path = self.root / "volume.npy"
        np.save(volume_path, np.arange(192, dtype=np.float32).reshape(4, 6, 8) + 1)
        (self.root / "meta_data.json").write_text(json.dumps({
            "scanner": {"sVoxel": [4, 2, 1], "offOrigin": [1, -2, .5]},
            "reconstruction": {"coordinate_mode": "gaussian"}}))
        points = []
        for name, module in (("old", self.old_init), ("new", init)):
            output = self.root / f"{name}.npy"
            with forbid_optional_imports(), contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                sys, "argv", ["init_from_fbp.py", "--volume", str(volume_path),
                              "--output", str(output), "--n_points", "20", "--seed", "0"]):
                module.main()
            points.append(np.load(output))
        np.testing.assert_array_equal(points[0], points[1])

    def test_default_nifti_export_matches_original(self):
        volume = np.arange(192, dtype=np.float32).reshape(4, 6, 8) - 2
        scanner = {"sVoxel": [.01, .02, .03]}
        exports = []
        for module in (self.old_init, init):
            sitk = ModuleType("SimpleITK")
            image = mock.Mock()
            sitk.GetImageFromArray = mock.Mock(return_value=image)
            sitk.WriteImage = mock.Mock()
            with forbid_optional_imports(), mock.patch.dict(sys.modules, {"SimpleITK": sitk}):
                module.save_fbp_nifti(volume, self.root / "preview.nii.gz", scanner)
            exports.append((sitk.GetImageFromArray.call_args.args[0],
                            image.SetSpacing.call_args.args[0]))
        np.testing.assert_array_equal(exports[0][0], exports[1][0])
        self.assertEqual(exports[0][1], exports[1][1])

    def test_gaussian_nifti_export_requires_explicit_mode(self):
        sitk = ModuleType("SimpleITK")
        image = mock.Mock()
        sitk.GetImageFromArray = mock.Mock(return_value=image)
        sitk.WriteImage = mock.Mock()
        with mock.patch.dict(sys.modules, {"SimpleITK": sitk}):
            init.save_fbp_nifti(np.ones((4, 6, 8)), self.root / "preview.nii.gz",
                                {"sVoxel": [.01, .02, .03]}, coordinate_mode="gaussian")
        np.testing.assert_allclose(image.SetSpacing.call_args.args[0],
                                   np.array([.01, .02, .03]) * (2 / .03) / [4, 6, 8])

    def test_each_feature_requires_its_own_explicit_option(self):
        source = self.root / "input"
        source.mkdir()
        tifffile.imwrite(source / "Tomo_0.tiff", np.ones((8, 12), dtype=np.float32))
        for auto in (False, True):
            for gaussian in (False, True):
                command = ["--input_dir", str(source), "--output_dir", str(self.root / "out"),
                           "--model_path", str(self.root / "model"), "--init_source", "train_fbp"]
                if auto:
                    command += ["--fbp_sVoxel", "auto", "--fbp_pixel_size", "0.00004",
                                "--fbp_pixel_subsample", "1"]
                if gaussian:
                    command += ["--fbp_coordinate_mode", "gaussian"]
                args = pipeline.resolve_args(pipeline.build_parser().parse_args(command))
                pipeline.validate_args(args)
                pipeline.resolve_pipeline_svoxel(args)
                fbp = pipeline.build_fbp_command(args, None, [0, 0])
                initial = pipeline.build_init_command(args, self.root / "init.npy")
                with self.subTest(auto=auto, gaussian=gaussian):
                    self.assertEqual("--coordinate_mode" in fbp, gaussian)
                    self.assertEqual("--coordinate_mode" in initial, gaussian)
                    self.assertEqual(fbp[fbp.index("--sVoxel") + 1] == "auto", auto)
                    self.assertEqual(hasattr(args, "fbp_sVoxel_resolution"), auto)


if __name__ == "__main__":
    unittest.main()
