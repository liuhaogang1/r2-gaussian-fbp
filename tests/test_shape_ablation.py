"""Six-case ablation wiring, read-only reuse verification and batch execution."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_pipeline as pipeline  # noqa: E402
from r2_gaussian.utils.artifact_utils import file_identity  # noqa: E402
from scripts import run_shape_ablation as ablation  # noqa: E402
from scripts import run_shape_grid as batch  # noqa: E402
from scripts import shape_grid_images as images  # noqa: E402
import test_shape_grid as grid_fixtures  # noqa: E402


class ShapeAblationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shape_ablation_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw input"
        self.raw.mkdir()
        (self.raw / "Tomo_001.tiff").touch()
        (self.raw / "Flat_001.tiff").touch()
        (self.raw / "config.txt").write_text("acquisition fixture", encoding="utf-8")

    def args(self, *extra):
        return ablation.build_parser().parse_args([
            "--input_dir", str(self.raw), "--config", str(self.raw / "config.txt"),
            "--output_dir", str(self.root / "output"), "--model_path", str(self.root / "models"),
            "--needle_threshold", "8", "--sheet_threshold", "12",
            "--lambda_needle", "0.02", "--lambda_sheet", "0.07", *extra,
        ])

    def test_six_cases_apply_independent_explicit_parameters_and_correct_switches(self):
        args = self.args()
        ablation.validate_args(args)
        plan = ablation.build_plan(args)
        expected = [
            (False, False, False, 0.0, 0.0, "none"),
            (True, True, False, 0.02, 0.0, "none"),
            (True, False, True, 0.0, 0.07, "none"),
            (True, True, True, 0.02, 0.07, "none"),
            (True, True, True, 0.0, 0.0, "repair"),
            (True, True, True, 0.02, 0.07, "repair"),
        ]
        self.assertEqual(len(plan["cases"]), 6)
        self.assertEqual(plan["shared_command"][-2:], ["--stop_after", "init"])
        reference = None
        allowed = {"model_path", "train_shape_enabled", "train_shape_needle_enabled",
                   "train_shape_sheet_enabled", "train_lambda_shape_needle", "train_lambda_shape_sheet",
                   "train_shape_hard_action"}
        for case, settings in zip(plan["cases"], expected):
            actual = tuple(case[k] for k in (
                "shape_enabled", "needle_enabled", "sheet_enabled", "lambda_needle", "lambda_sheet", "hard_action"))
            self.assertEqual(actual, settings)
            values = case["resolved_arguments"]
            self.assertEqual(values["train_shape_needle_threshold"], 8)
            self.assertEqual(values["train_shape_sheet_threshold"], 12)
            self.assertEqual(values["train_shape_hard_needle_ratio"], 50)
            self.assertEqual(values["train_shape_hard_sheet_ratio"], 50)
            self.assertEqual(values["train_shape_max_hard_actions_per_check"], 32)
            self.assertEqual(values["train_iterations"], 30000)
            cmd = case["train_command"]
            self.assertEqual("--shape_enabled" in cmd, settings[0])
            self.assertEqual("--shape_needle_enabled" in cmd, settings[1])
            self.assertEqual("--shape_sheet_enabled" in cmd, settings[2])
            self.assertEqual(cmd[cmd.index("--ply_path") + 1], plan["init_path"])
            fixed = {k: v for k, v in values.items() if k not in allowed}
            if reference is None:
                reference = fixed
            self.assertEqual(fixed, reference)

    def test_missing_invalid_parameters_and_partial_reuse_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ablation.build_parser().parse_args([])
        for flag, value in (("--needle_threshold", "1"), ("--sheet_threshold", "nan"),
                            ("--lambda_needle", "-0.1"), ("--lambda_sheet", "inf")):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                ablation.validate_args(self.args(flag, value))
        with self.assertRaisesRegex(ValueError, "together"):
            ablation.validate_args(self.args("--shared_dataset_dir", str(self.root / "old")))

    def test_dry_run_launches_nothing_and_writes_nothing(self):
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(batch, "run_logged") as run:
            self.assertEqual(ablation.main(self.args("--dry_run")), 0)
        run.assert_not_called()
        self.assertFalse((self.root / "output").exists())
        self.assertFalse((self.root / "models").exists())

    def create_reuse_fixture(self):
        dataset = self.root / "old/shared/dataset"
        init = self.root / "old/shared/init"
        dataset.mkdir(parents=True)
        init.mkdir()
        points = init / "init_points.npy"
        # Artifact verification tests use small reference volumes; their saved
        # provenance reports the original production grid and sampling count.
        projection = dataset / "proj.npy"
        np.save(projection, np.ones((2, 2), dtype=np.float32))
        for filename in ("vol_fbp.npy", "vol_train_fbp.npy"):
            np.save(dataset / filename, np.ones((2, 2, 2), dtype=np.float32))
        np.save(init / "vol_init_denoised.npy", np.ones((2, 2, 2), dtype=np.float32))
        np.save(points, np.ones((100000, 4), dtype=np.float32))
        center_path = self.root / "old/shared/center.json"
        center_path.write_text("{}", encoding="utf-8")
        metadata_path = dataset / "meta_data.json"
        metadata_path.write_text(json.dumps({
            "scanner": {"mode": "parallel", "nVoxel": [256]*3, "sVoxel": [0.01]*3},
            "vol": "vol_fbp.npy", "proj_train": [{"file_path": "proj.npy", "angle": i*0.01} for i in range(360)],
            "proj_test": [],
        }), encoding="utf-8")
        report = {
            "input": {"volume_path": str(dataset / "vol_train_fbp.npy"),
                      "volume_sha256": file_identity(dataset / "vol_train_fbp.npy")["sha256"],
                      "metadata_path": str(metadata_path), "metadata_sha256": file_identity(metadata_path)["sha256"]},
            "point_output": str(points), "point_sha256": file_identity(points)["sha256"],
            "volume_output": str(init / "vol_init_denoised.npy"),
            "volume_sha256": file_identity(init / "vol_init_denoised.npy")["sha256"],
            "sigma_voxels": 0.8, "normalization_percentile": 99.5, "density_threshold_normalized": 0.25,
            "density_rescale": 0.2, "seed": 0, "sampled_count": 100000, "shape_xyz": [256]*3,
        }
        (init / "denoise_report.json").write_text(json.dumps(report), encoding="utf-8")
        args = self.args()
        ablation.validate_args(args)
        resolved = pipeline.resolve_args(pipeline.build_parser().parse_args(ablation.original_options(args)))
        resolved.output_dir = dataset
        resolved.init_output = points
        resolved.init_denoise_output_dir = init
        resolved.center_output = center_path
        resolved.fbp_sVoxel_requested = "auto"
        resolved.fbp_sVoxel = [0.01]*3
        provenance = {
            "resolved_arguments": pipeline._jsonable(vars(resolved)),
            "stage_status": dict(center="success", fbp="success", init="success"),
            "stage_records": {
                "center": {"artifacts": [file_identity(center_path)]},
                "fbp": {"artifacts": [file_identity(p) for p in dataset.iterdir() if p.name != "vol_train_fbp.npy"]},
                "init": {"artifacts": [file_identity(p) for p in init.iterdir()]},
            },
        }
        (init / "pipeline_manifest.json").write_text(json.dumps(provenance), encoding="utf-8")
        outer = self.root / "old/shape_grid_manifest.json"
        outer.write_text(json.dumps({"configuration_identity": [file_identity(args.config)]}), encoding="utf-8")
        return dataset, points

    def reuse_args(self, dataset, points, *extra):
        return self.args("--shared_dataset_dir", str(dataset), "--shared_init_path", str(points), *extra)

    def test_verified_reuse_changes_all_case_sources_without_writing_old_inputs(self):
        dataset, points = self.create_reuse_fixture()
        before = {str(p): file_identity(p) for p in (self.root / "old").rglob("*") if p.is_file()}
        args = self.reuse_args(dataset, points)
        ablation.validate_args(args)
        plan = ablation.build_plan(args)
        self.assertIsNone(plan["shared_command"])
        self.assertTrue(plan["shared_validation"]["configuration_hash_verified"])
        for case in plan["cases"]:
            cmd = case["train_command"]
            self.assertEqual(cmd[cmd.index("-s") + 1], str(dataset))
            self.assertEqual(cmd[cmd.index("--ply_path") + 1], str(points))
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(batch.subprocess, "run") as run:
            self.assertEqual(ablation.main(self.reuse_args(dataset, points, "--dry_run")), 0)
        run.assert_not_called()
        self.assertEqual(before, {str(p): file_identity(p) for p in (self.root / "old").rglob("*") if p.is_file()})

    def test_reuse_rejects_modified_artifacts(self):
        dataset, points = self.create_reuse_fixture()
        np.save(points, np.full((100000, 4), 2, dtype=np.float32))
        args = self.reuse_args(dataset, points)
        ablation.validate_args(args)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ablation.build_plan(args)
        self.assertFalse(args.output_dir.exists())

    def test_reuse_rejects_mismatched_preprocessing_settings(self):
        dataset, points = self.create_reuse_fixture()
        path = points.parent / "pipeline_manifest.json"
        provenance = json.loads(path.read_text())
        provenance["resolved_arguments"]["init_density_thresh"] = 0.1
        path.write_text(json.dumps(provenance))
        args = self.reuse_args(dataset, points)
        ablation.validate_args(args)
        with self.assertRaisesRegex(ValueError, "init_density_thresh"):
            ablation.build_plan(args)

    def test_reuse_rejects_different_acquisition_config_content(self):
        dataset, points = self.create_reuse_fixture()
        (self.raw / "config.txt").write_text("changed acquisition fixture")
        args = self.reuse_args(dataset, points)
        ablation.validate_args(args)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ablation.build_plan(args)

    def fake_process(self, command, **kwargs):
        return grid_fixtures.ShapeGridTests.fake_process(self, command, **kwargs)

    def fake_render(self, cases, metadata_path, output_dir):
        self.plotted = list(cases)
        return {"cases": [{"case": c["case"], "png": str(output_dir / (c["case"] + ".png"))} for c in cases]}

    def run_mock_batch(self, args, fail_case=None):
        self.scripts, self.configs, self.plotted = [], [], []
        self.fail_case = fail_case
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(batch.subprocess, "run", side_effect=self.fake_process), \
                mock.patch.object(batch, "render_grid", side_effect=self.fake_render):
            code = ablation.main(args)
        manifest = json.loads((self.root / "output" / ablation.MANIFEST).read_text())
        return code, manifest

    def test_full_six_case_run_prepares_once_and_preserves_baseline_and_hard_only(self):
        code, manifest = self.run_mock_batch(self.args())
        self.assertEqual(code, 0)
        self.assertEqual(self.scripts.count("run_pipeline.py"), 1)
        self.assertEqual(self.scripts.count("train.py"), 6)
        self.assertEqual(self.scripts.count("test.py"), 6)
        self.assertEqual(manifest["plot_status"], "completed")
        self.assertEqual(len(self.plotted), 6)
        self.assertEqual(len({c["ply_path"] for c in self.configs}), 1)
        self.assertFalse(self.configs[0]["shape_enabled"])
        self.assertTrue(self.configs[4]["shape_enabled"])
        self.assertEqual(self.configs[4]["shape_hard_action"], "repair")
        self.assertEqual(self.configs[4]["lambda_shape_needle"], 0)
        self.assertEqual(self.configs[4]["lambda_shape_sheet"], 0)
        self.assertIn("lambda_needle", (self.root / "output/summary.csv").read_text(encoding="utf-8-sig"))

    def test_reused_run_skips_all_shared_stages_and_runs_six_train_test_pairs(self):
        dataset, points = self.create_reuse_fixture()
        before = file_identity(points)
        code, manifest = self.run_mock_batch(self.reuse_args(dataset, points))
        self.assertEqual(code, 0)
        self.assertNotIn("run_pipeline.py", self.scripts)
        self.assertEqual(self.scripts.count("train.py"), 6)
        self.assertEqual(self.scripts.count("test.py"), 6)
        self.assertEqual(manifest["shared_status"], "reused")
        self.assertTrue(all(c["ply_path"] == str(points) for c in self.configs))
        self.assertEqual(before, file_identity(points))

    def test_failed_trial_does_not_contaminate_images_or_block_later_trials(self):
        code, manifest = self.run_mock_batch(self.args(), "A1_needle")
        self.assertEqual(code, 1)
        self.assertEqual(len(self.plotted), 5)
        self.assertEqual(manifest["plot_status"], "partial")
        self.assertEqual(self.scripts.count("train.py"), 6)
        self.assertNotIn("A1_needle", [c["case"] for c in self.plotted])

    def test_plot_only_uses_saved_parameters_and_launches_no_training(self):
        code, _ = self.run_mock_batch(self.args())
        self.assertEqual(code, 0)
        (self.raw / "config.txt").unlink()
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(batch.subprocess, "run") as run, \
                mock.patch.object(batch, "render_grid", side_effect=self.fake_render):
            self.assertEqual(ablation.main(self.args("--plot_only")), 0)
        run.assert_not_called()
        with self.assertRaisesRegex(ValueError, "needle_threshold"):
            ablation.main(self.args("--plot_only", "--needle_threshold", "20"))


class AblationImageTests(unittest.TestCase):
    def test_ablation_montage_labels_actual_branch_settings_and_contains_nine_slices(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        with tempfile.TemporaryDirectory(prefix="ablation_image_") as temporary:
            root = Path(temporary)
            path = root / "volume.npy"
            np.save(path, np.arange(192, dtype=np.float32).reshape(4, 6, 8))
            case = dict(case="A4_hard_repair", iteration=30000, shape_enabled=True,
                        needle_enabled=True, sheet_enabled=True, needle_threshold=8, sheet_threshold=12,
                        lambda_needle=0, lambda_sheet=0, hard_action="repair", volume_path=str(path))
            titles = []
            original = Figure.suptitle

            def capture(fig, text, *args, **kwargs):
                titles.append(text)
                return original(fig, text, *args, **kwargs)

            with mock.patch.object(Figure, "suptitle", new=capture):
                report = images.render_case(case, {"sVoxel": [4, 12, 24]}, root / "images",
                                            (0, 191), (4, 6, 8))
            self.assertIn("hard action = repair", titles[0])
            self.assertIn("threshold needle/sheet = 8/12", titles[0])
            self.assertIn("lambda needle/sheet = 0/0", titles[0])
            with np.load(report["slices"]) as slices:
                self.assertEqual(len(slices.files), 9)
            self.assertTrue(Path(report["png"]).is_file())


if __name__ == "__main__":
    unittest.main()
