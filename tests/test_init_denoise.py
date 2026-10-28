"""CPU checks for opt-in initialization denoising and pipeline routing."""

import builtins
import contextlib
import gzip
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

import numpy as np
from scipy.ndimage import gaussian_filter
import yaml

import init_from_fbp as init
import run_pipeline as pipeline
from r2_gaussian.utils.geometry_utils import initialization_positions
from scripts import init_denoise as denoise


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "8dbc3a5ba17d56d24dc5f83058e3af27bcaaa0cf"


class InitDenoiseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.volume_path = self.dataset / "vol_fbp.npy"
        self.model = self.root / "model"
        self.artifacts = self.model / "init_denoise"
        self.shape = (13, 15, 17)
        rng = np.random.default_rng(9)
        self.volume = rng.uniform(0, 0.08, self.shape).astype(np.float32)
        self.volume[3:10, 4:12, 3:14] += 0.7
        self.volume[1, 1, 1] = 2.5
        np.save(self.volume_path, self.volume)
        self.scanner = {"nVoxel": list(self.shape), "sVoxel": [1.3, 3.0, 5.1],
                        "offOrigin": [0.1, -0.4, 0.9], "mode": "parallel"}
        self.metadata_path = self.dataset / "meta_data.json"
        self.metadata_path.write_text(json.dumps({"scanner": self.scanner}), encoding="utf-8")
        self.canonical = self.dataset / "init_dataset.npy"
        np.save(self.canonical, np.array([[99, 98, 97, 96]], dtype=np.float32))

    def init_args(self, extra=()):
        return ["init_from_fbp.py", "--volume", str(self.volume_path), "--n_points", "40",
                "--density_thresh", "0.10", "--density_rescale", "0.25", "--seed", "7", *extra]

    def run_init(self, extra=(), module=init, previews=False):
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(sys, "argv", self.init_args(extra)):
            if previews:
                module.main()
            else:
                with mock.patch.object(denoise, "save_previews", return_value=[0, 1]):
                    module.main()

    def pipeline_args(self, extra=()):
        return pipeline.resolve_args(pipeline.build_parser().parse_args([
            "--input_dir", str(self.root / "raw"), "--output_dir", str(self.dataset),
            "--model_path", str(self.model), "--init_denoise", "--start_from", "init",
            "--stop_after", "train", "--init_n_points", "40", "--init_density_thresh", "0.1",
            "--init_density_rescale", "0.25", "--no-init_visualize", "--no-init_visualize_slices", *extra]))

    def run_pipeline(self, args):
        calls = []

        def execute(command, **kwargs):
            calls.append(command)
            if Path(command[1]).name == "init_from_fbp.py":
                with mock.patch.object(sys, "argv", command[1:]):
                    init.main()
            return mock.Mock(returncode=0)

        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            pipeline.subprocess, "run", side_effect=execute
        ), mock.patch.object(denoise, "save_previews", return_value=[0, 1]):
            pipeline.main(args)
        return calls

    def test_disabled_matches_pinned_initializer_and_does_not_import_denoiser(self):
        source = subprocess.check_output(["git", "show", f"{BASELINE}:init_from_fbp.py"], cwd=ROOT).decode("utf-8")
        old = ModuleType("previous_initializer")
        old.__file__ = str(ROOT / "init_from_fbp.py")
        exec(compile(source, old.__file__, "exec"), old.__dict__)
        output = self.root / "legacy_points.npy"
        self.run_init(["--output", str(output)], module=old)
        expected = output.read_bytes()
        canonical_expected = self.canonical.read_bytes()
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name in {"scripts.init_denoise", "scipy.ndimage"}:
                raise AssertionError(f"Disabled feature imported {name}")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded):
            self.run_init(["--output", str(output)])
        self.assertEqual(output.read_bytes(), expected)
        self.assertEqual(self.canonical.read_bytes(), canonical_expected)
        self.assertFalse((self.dataset / "init_denoise").exists())

    def test_filter_sampling_densities_and_original_files(self):
        original_bytes = self.volume_path.read_bytes()
        canonical_bytes = self.canonical.read_bytes()
        self.run_init(["--denoise", "--denoise_output_dir", str(self.artifacts)])
        filtered = gaussian_filter(self.volume, sigma=0.8, mode="nearest", truncate=4.0)
        np.testing.assert_array_equal(np.load(self.artifacts / "vol_init_denoised.npy"), filtered)
        scale = np.percentile(filtered[filtered > 0], 99.5)
        normalized = np.clip(filtered / scale, 0, 1)
        indices = np.argwhere(normalized > 0.10)
        selected = indices[np.random.default_rng(7).choice(len(indices), 40, replace=False)]
        expected = np.column_stack((initialization_positions(selected, self.shape, self.scanner),
                                    normalized[tuple(selected.T)] * 0.25)).astype(np.float32)
        np.testing.assert_array_equal(np.load(self.artifacts / "init_points.npy"), expected)
        self.assertEqual(self.volume_path.read_bytes(), original_bytes)
        self.assertEqual(self.canonical.read_bytes(), canonical_bytes)
        report = json.loads((self.artifacts / "denoise_report.json").read_text())
        self.assertEqual(report["candidate_count"], len(indices))
        self.assertAlmostEqual(report["normalization_scale"], float(scale))
        self.assertEqual(report["sampled_count"], 40)
        self.assertAlmostEqual(report["density_threshold_before_normalization"], float(scale) * 0.1)

    def test_no_percentile_and_insufficient_candidates(self):
        self.run_init(["--denoise", "--normalize_percentile", "0"])
        report = json.loads((self.dataset / "init_denoise" / "denoise_report.json").read_text())
        self.assertEqual(report["normalization_scale"], 1.0)
        with self.assertRaisesRegex(ValueError, "Only 0 voxels"):
            self.run_init(["--denoise", "--density_thresh", "2"])

    def test_invalid_sigma_and_missing_metadata_fail_without_outputs(self):
        for sigma in ("0", "-0.1", "nan", "inf"):
            with self.subTest(sigma=sigma), self.assertRaisesRegex(ValueError, "sigma"):
                self.run_init(["--denoise", "--denoise_sigma", sigma])
        self.metadata_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "scanner metadata"):
            self.run_init(["--denoise"])
        self.assertFalse((self.dataset / "init_denoise").exists())

    def test_conflicting_outputs_are_rejected(self):
        original_bytes = self.volume_path.read_bytes()
        canonical_bytes = self.canonical.read_bytes()
        for path in (self.canonical, self.volume_path, self.metadata_path,
                     self.artifacts / "vol_init_denoised.npy", self.volume_path.with_suffix("")):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "overwrite|conflicts|suffix"):
                self.run_init(["--denoise", "--output", str(path),
                               "--denoise_output_dir", str(self.artifacts)])
        self.assertEqual(self.volume_path.read_bytes(), original_bytes)
        self.assertEqual(self.canonical.read_bytes(), canonical_bytes)

    def test_nifti_preserves_xyz_values_spacing_and_voxel_center_origin(self):
        self.artifacts.mkdir(parents=True)
        path = self.artifacts / "volume.nii.gz"
        info = denoise.save_nifti_xyz(self.volume, self.scanner, path)
        # Re-export is supported for this experiment's own artifacts.
        denoise.save_nifti_xyz(self.volume, self.scanner, path)
        with gzip.open(path, "rb") as stream:
            payload = stream.read()
        self.assertEqual(struct.unpack_from("<8h", payload, 40), (3, *self.shape, 1, 1, 1, 1))
        data = np.frombuffer(payload, dtype="<f4", offset=352).reshape(self.shape, order="F")
        np.testing.assert_array_equal(data, self.volume)
        spacing = np.asarray(self.scanner["sVoxel"]) / self.shape
        origin = np.asarray(self.scanner["offOrigin"]) - np.asarray(self.scanner["sVoxel"]) / 2 + spacing / 2
        np.testing.assert_allclose(struct.unpack_from("<8f", payload, 76)[1:4], spacing)
        sform = np.array([struct.unpack_from("<4f", payload, offset) for offset in (280, 296, 312)])
        np.testing.assert_allclose(sform, np.column_stack((np.diag([-spacing[0], -spacing[1], spacing[2]]),
                                                        origin * [-1, -1, 1])), atol=1e-6)
        np.testing.assert_allclose(info["origin_lps"], origin)

    def test_real_previews_and_optional_nifti_are_written(self):
        self.run_init(["--denoise", "--denoise_output_dir", str(self.artifacts), "--denoise_save_nii"], previews=True)
        for axis in "xyz":
            for prefix in ("comparison", "candidate_mask"):
                path = self.artifacts / f"{prefix}_center_{axis}.png"
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue((self.artifacts / "vol_init_denoised.nii.gz").exists())

    def test_pipeline_routes_new_points_and_preserves_baseline_configuration(self):
        legacy_manifest = self.dataset / "pipeline_manifest.json"
        legacy_config = self.dataset / ".pipeline_train_config.yml"
        legacy_manifest.write_text("original manifest")
        legacy_config.write_text("original config")
        canonical_bytes = self.canonical.read_bytes()
        args = self.pipeline_args()
        calls = self.run_pipeline(args)
        train = next(c for c in calls if Path(c[1]).name == "train.py")
        points = str(self.artifacts / "init_points.npy")
        self.assertEqual(train[train.index("--ply_path") + 1], points)
        config_path = Path(train[train.index("--config") + 1])
        self.assertEqual(yaml.safe_load(config_path.read_text())["ply_path"], points)
        self.assertEqual(legacy_manifest.read_text(), "original manifest")
        self.assertEqual(legacy_config.read_text(), "original config")
        self.assertEqual(self.canonical.read_bytes(), canonical_bytes)
        manifest = json.loads((self.artifacts / "pipeline_manifest.json").read_text())
        self.assertEqual(manifest["stage_status"]["init"], "success")
        self.assertEqual(manifest["stage_status"]["train"], "success")

    def test_all_initialization_sources_are_denoised(self):
        train_volume = self.dataset / "vol_train_fbp.npy"
        np.save(train_volume, self.volume * 0.7)
        custom_volume = self.dataset / "custom.npy"
        np.save(custom_volume, self.volume * 0.4)
        for source, volume in (("full_fbp", self.volume_path), ("train_fbp", train_volume), ("custom", custom_volume)):
            with self.subTest(source=source):
                extra = ["--init_source", source, "--model_path", str(self.root / source)]
                if source == "custom":
                    extra += ["--init_volume", str(custom_volume)]
                args = self.pipeline_args(extra)
                self.run_pipeline(args)
                report = json.loads((args.init_denoise_output_dir / "denoise_report.json").read_text())
                self.assertEqual(report["input"]["volume_path"], str(volume))
                np.testing.assert_array_equal(np.load(args.init_denoise_output_dir / "vol_init_denoised.npy"),
                                              gaussian_filter(np.load(volume), 0.8, mode="nearest", truncate=4.0))

    def test_resume_repeated_and_parameter_changes_are_checked(self):
        self.run_pipeline(self.pipeline_args())
        for _ in range(2):
            self.assertEqual(self.run_pipeline(self.pipeline_args(["--resume"])), [])
        with self.assertRaisesRegex(ValueError, "denoising disabled"):
            self.run_pipeline(self.pipeline_args(["--resume", "--no-init_denoise"]))
        for option in (["--init_denoise_sigma", "0.5"], ["--init_density_thresh", "0.15"],
                       ["--init_density_rescale", "0.20"], ["--seed", "8"],
                       ["--output_dir", str(self.root / "another_dataset")]):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "settings changed"):
                self.run_pipeline(self.pipeline_args(["--resume", *option]))

    def test_resume_rejects_replaced_report(self):
        self.run_pipeline(self.pipeline_args())
        path = self.artifacts / "denoise_report.json"
        report = json.loads(path.read_text())
        report["density_rescale"] = 0.20
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "report changed"):
            self.run_pipeline(self.pipeline_args(["--resume"]))

    def test_custom_artifact_directory_preserves_resume_checks(self):
        directory = self.root / "custom_artifacts"
        self.run_pipeline(self.pipeline_args(["--init_denoise_output_dir", str(directory)]))
        self.assertTrue((directory / "init_points.npy").exists())
        self.assertTrue((self.artifacts / "pipeline_manifest.json").exists())
        self.assertEqual(self.run_pipeline(self.pipeline_args([
            "--init_denoise_output_dir", str(directory), "--resume"])), [])
        with self.assertRaisesRegex(ValueError, "settings changed"):
            self.run_pipeline(self.pipeline_args(["--resume"]))

    def test_resume_rejects_changed_inputs_and_point_cloud(self):
        self.run_pipeline(self.pipeline_args())
        original = self.volume_path.read_bytes()
        np.save(self.volume_path, self.volume * 2)
        with self.assertRaisesRegex(ValueError, "input or metadata changed"):
            self.run_pipeline(self.pipeline_args(["--resume"]))
        self.volume_path.write_bytes(original)
        metadata = self.metadata_path.read_bytes()
        self.metadata_path.write_text(json.dumps({"scanner": self.scanner, "changed": True}))
        with self.assertRaisesRegex(ValueError, "input or metadata changed"):
            self.run_pipeline(self.pipeline_args(["--resume"]))
        self.metadata_path.write_bytes(metadata)
        np.save(self.artifacts / "init_points.npy", np.zeros((40, 4)))
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            self.run_pipeline(self.pipeline_args(["--resume"]))

    def test_pipeline_rejects_skipped_init_checkpoint_or_external_points(self):
        for option, message in ((["--skip_init"], "requires the init stage"),
                                (["--start_from", "train"], "requires the init stage"),
                                (["--train_start_checkpoint", "old.pth"], "checkpoint"),
                                (["--train_ply_path", "old.npy"], "point cloud")):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, message):
                pipeline.validate_args(self.pipeline_args(option))
        self.model.mkdir()
        (self.model / "cfg_args").write_text("old model")
        for extra in ([], ["--resume"]):
            with self.assertRaisesRegex(ValueError, "new model directory"):
                self.run_pipeline(self.pipeline_args(extra))

    def test_yaml_cli_override_and_explicit_disable(self):
        path = self.root / "pipeline.yml"
        path.write_text(yaml.safe_dump({"init": {"denoise": True, "denoise_sigma": 0.5,
                                                "denoise_save_nii": True}}))
        args = self.pipeline_args(["--pipeline_config", str(path), "--init_denoise_sigma", "0.8"])
        self.assertEqual(args.init_denoise_sigma, 0.8)
        self.assertTrue(args.init_denoise_save_nii)
        disabled = self.pipeline_args(["--pipeline_config", str(path), "--no-init_denoise"])
        self.assertFalse(hasattr(disabled, "init_denoise"))
        command = pipeline.build_init_command(disabled, self.root / "init.npy")
        self.assertNotIn("--denoise", command)


if __name__ == "__main__":
    unittest.main()
