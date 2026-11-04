"""CPU coverage for the nine-case wrapper, failure handling and XYZ montages."""

import contextlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_shape_grid as grid  # noqa: E402
from scripts import shape_grid_images as images  # noqa: E402


class ShapeGridTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="shape_grid_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw input"
        self.raw.mkdir()
        (self.raw / "Tomo_001.tiff").touch()
        (self.raw / "Flat_001.tiff").touch()
        (self.raw / "config.txt").write_text("acquisition fixture", encoding="utf-8")

    def args(self, *extra):
        return grid.build_parser().parse_args([
            "--input_dir", str(self.raw), "--output_dir", str(self.root / "output"),
            "--model_path", str(self.root / "models"), "--config", str(self.raw / "config.txt"),
            *extra,
        ])

    def test_nine_cases_share_data_init_and_preserve_original_settings(self):
        args = self.args()
        grid.validate_args(args)
        plan = grid.build_plan(args)
        self.assertEqual(len(plan["cases"]), 9)
        self.assertEqual({(c["threshold"], c["weight"]) for c in plan["cases"]},
                         {(t, w) for t in (5, 10, 20) for w in (0.01, 0.05, 0.1)})
        shared = plan["shared_command"]
        self.assertEqual(shared[-2:], ["--stop_after", "init"])
        expected = {
            "fbp_flat_field": True, "fbp_dark_value": 0.0, "fbp_flat_eps": 1e-6,
            "stripe_correction": {"snr": 3.0, "la_size": 61, "sm_size": 21, "dim": 1},
            "fbp_input_type": "transmission", "fbp_log_eps": 1e-6, "fbp_mode": "parallel",
            "fbp_shift_v": 0, "fbp_pixel_subsample": 1, "fbp_recon_pixel_subsample": 1,
            "fbp_resize_order": 1, "fbp_projection_scale": 0.125, "fbp_pixel_size": 0.00004,
            "fbp_nVoxel": [256, 256, 256], "fbp_sVoxel": "auto", "fbp_offOrigin": [0, 0, 0],
            "fbp_DSD": 7, "fbp_DSO": 5, "fbp_accuracy": 0.5, "fbp_filter": "hann",
            "center_method": "pair", "center_pixel_size": 0.00004,
            "fbp_n_train": 360, "fbp_n_test": 0, "init_source": "train_fbp",
            "init_denoise": True, "init_denoise_sigma": 0.8, "init_denoise_save_nii": True,
            "init_n_points": 100000, "init_density_thresh": 0.25, "init_density_rescale": 0.2,
            "init_normalize_percentile": 99.5, "seed": 0, "init_seed": 0,
            "train_pose_enabled": True, "train_parallel_optimize_angle": True,
            "train_parallel_optimize_global_detector_shift": True,
            "train_parallel_optimize_per_view_detector_shift": True,
            "train_parallel_optimize_beam_tilt": True, "train_parallel_optimize_detector_roll": True,
            "train_shape_enabled": True, "train_shape_needle_enabled": True, "train_shape_sheet_enabled": True,
            "train_shape_start_iter": 500, "train_shape_end_iter": 30000,
            "train_shape_ramp_iters": 1000, "train_shape_hard_action": "none",
            "train_max_num_gaussians": 300000, "train_iterations": 30000,
            "test_iteration": -1, "test_skip_render_test": True, "test_skip_recon": False,
        }
        for case in plan["cases"]:
            values = case["resolved_arguments"]
            for key, value in expected.items():
                self.assertEqual(values[key], value, key)
            self.assertEqual(values["train_shape_needle_threshold"], case["threshold"])
            self.assertEqual(values["train_shape_sheet_threshold"], case["threshold"])
            self.assertEqual(values["train_lambda_shape_needle"], case["weight"])
            self.assertEqual(values["train_lambda_shape_sheet"], case["weight"])
            cmd = case["train_command"]
            self.assertEqual(cmd[cmd.index("--ply_path") + 1], plan["init_path"])
            self.assertEqual(cmd[cmd.index("-s") + 1], str(Path(plan["metadata_path"]).parent))
            self.assertIn("--skip_render_test", case["test_command"])
            self.assertNotIn("--skip_recon", case["test_command"])
            self.assertNotIn("--seed", cmd)  # preserve the existing fixed training seed
        self.assertEqual(len({c["model_path"] for c in plan["cases"]}), 9)

    def test_dry_run_writes_nothing_and_existing_outputs_are_protected(self):
        args = self.args("--dry_run")
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(grid.subprocess, "run") as run:
            self.assertEqual(grid.main(args), 0)
        run.assert_not_called()
        self.assertFalse(args.output_dir.exists())
        self.assertFalse(args.model_path.exists())
        args.output_dir.mkdir()
        keep = args.output_dir / "previous_result"
        keep.touch()
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            grid.main(self.args())
        self.assertTrue(keep.exists())
        with self.assertRaisesRegex(ValueError, "raw input tree"):
            grid.validate_args(self.args("--output_dir", str(self.raw / "result")))

    def fake_process(self, command, **kwargs):
        script = Path(command[2]).name
        self.scripts.append(script)
        if script == "run_pipeline.py":
            target = Path(command[command.index("--output_dir") + 1])
            target.mkdir(parents=True)
            (target / "meta_data.json").write_text(json.dumps({"scanner": {
                "nVoxel": [4, 6, 8], "sVoxel": [4, 12, 24], "offOrigin": [0, 0, 0]}}))
            points = Path(command[command.index("--init_output") + 1])
            points.parent.mkdir(parents=True)
            np.save(points, np.ones((2, 4)))
            return subprocess.CompletedProcess(command, 0)
        model = Path(command[command.index("-m") + 1])
        if script == "train.py":
            if model.name == self.fail_case:
                return subprocess.CompletedProcess(command, 9)
            cfg = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
            self.configs.append(cfg)
            path = model / "point_cloud/iteration_30000/point_cloud.pickle"
            path.parent.mkdir(parents=True)
            with path.open("wb") as stream:
                pickle.dump({"scale": np.log([[1, 2, 30], [1, 20, 30]]),
                             "density": np.array([[1.0], [2.0]]), "scale_bound": None}, stream)
        elif script == "test.py":
            target = model / "test/iter_30000"
            target.mkdir(parents=True)
            np.save(target / "vol_pred.npy", np.arange(192, dtype=np.float32).reshape(4, 6, 8))
            (target / "evaluation_status.json").write_text(json.dumps({
                "status": "completed", "iteration": 30000, "n_test": 0}))
            (target / "eval2d_render_train.yml").write_text("psnr_2d: 30.0\nssim_2d: 0.9\n")
            (target / "eval3d.yml").write_text("psnr_3d: 25.0\nssim_3d: 0.8\n")
        return subprocess.CompletedProcess(command, 0)

    def fake_render(self, cases, metadata_path, output_dir):
        self.plotted = list(cases)
        return {"cases": [{"case": c["case"], "png": str(output_dir / (c["case"] + ".png"))} for c in cases]}

    def run_mock_batch(self, fail_case=None, *extra):
        self.scripts, self.configs, self.plotted = [], [], []
        self.fail_case = fail_case
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(grid.subprocess, "run", side_effect=self.fake_process), \
                mock.patch.object(grid, "render_grid", side_effect=self.fake_render):
            code = grid.main(self.args(*extra))
        manifest = json.loads((self.root / "output/shape_grid_manifest.json").read_text())
        return code, manifest

    def test_full_batch_runs_shared_once_and_nine_independent_train_test_pairs(self):
        code, manifest = self.run_mock_batch()
        self.assertEqual(code, 0)
        self.assertEqual(self.scripts.count("run_pipeline.py"), 1)
        self.assertEqual(self.scripts.count("train.py"), 9)
        self.assertEqual(self.scripts.count("test.py"), 9)
        self.assertEqual(len(self.plotted), 9)
        self.assertEqual(manifest["plot_status"], "completed")
        self.assertEqual(len({c["ply_path"] for c in self.configs}), 1)
        self.assertEqual({c["iterations"] for c in self.configs}, {30000})
        for cfg in self.configs:
            self.assertEqual(cfg["shape_needle_threshold"], cfg["shape_sheet_threshold"])
            self.assertEqual(cfg["lambda_shape_needle"], cfg["lambda_shape_sheet"])
        self.assertIn("psnr_3d_vs_fbp", (self.root / "output/summary.csv").read_text(encoding="utf-8-sig"))

    def test_failed_case_is_excluded_from_images_and_later_trials_continue(self):
        code, manifest = self.run_mock_batch("T5_L0p05")
        self.assertEqual(code, 1)
        self.assertEqual(self.scripts.count("train.py"), 9)
        self.assertEqual(self.scripts.count("test.py"), 8)
        self.assertEqual(len(self.plotted), 8)
        self.assertEqual(manifest["plot_status"], "partial")
        self.assertEqual(manifest["cases"][1]["status"], "failed")

    def test_fail_fast_marks_pending_cases_skipped(self):
        code, manifest = self.run_mock_batch("T5_L0p01", "--fail_fast")
        self.assertEqual(code, 1)
        self.assertEqual(self.scripts.count("train.py"), 1)
        self.assertEqual(sum(c["status"] == "skipped" for c in manifest["cases"]), 8)

    def test_plot_only_never_launches_preprocessing_training_or_testing(self):
        code, _ = self.run_mock_batch()
        self.assertEqual(code, 0)
        # Plot-only does not need the raw inputs to remain mounted.
        (self.raw / "config.txt").unlink()
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(grid.subprocess, "run") as run, \
                mock.patch.object(grid, "render_grid", side_effect=self.fake_render):
            self.assertEqual(grid.main(self.args("--plot_only")), 0)
        run.assert_not_called()
        self.assertEqual(len(self.plotted), 9)

    def test_perfect_fit_metrics_remain_valid_in_json_summary(self):
        code, manifest = self.run_mock_batch()
        self.assertEqual(code, 0)
        case = manifest["cases"][0]
        evaluation = Path(case["volume_path"]).parent / "eval2d_render_train.yml"
        evaluation.write_text("psnr_2d: .inf\nssim_2d: 1.0\n")
        case["metrics"] = grid.collect_metrics(case)
        self.assertEqual(case["metrics"]["train_psnr"], "inf")
        grid.save_progress(self.args(), manifest)

    def test_shape_statistics_activate_bounded_scales_and_density(self):
        path = self.root / "gaussian.pickle"
        target = np.array([[1.0, 2.0, 30.0], [1.0, 20.0, 30.0]])
        unit = (target - 0.5) / (100.0 - 0.5)
        with path.open("wb") as stream:
            pickle.dump({"scale": np.log(unit / (1 - unit)), "scale_bound": [0.5, 100.0],
                         "density": np.zeros((2, 1))}, stream)
        stats = grid.shape_statistics(path)
        self.assertEqual(stats["num_gaussians"], 2)
        self.assertEqual(stats["needle_gt10_fraction"], 0.5)
        self.assertEqual(stats["sheet_gt10_fraction"], 0.5)
        self.assertAlmostEqual(stats["needle_gt10_mass_fraction"], 60 / 660)


class ShapeGridImageTests(unittest.TestCase):
    def test_nine_panels_have_correct_axes_indices_and_shared_window(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.axes import Axes
        with tempfile.TemporaryDirectory(prefix="shape_grid_images_") as temporary:
            root = Path(temporary)
            shape = (4, 6, 8)
            base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            cases = []
            for i in range(2):
                path = root / f"vol{i}.npy"
                np.save(path, base * (i + 1))
                cases.append({"case": f"case{i}", "threshold": 5 + i * 5,
                              "weight": 0.05, "iteration": 30000, "volume_path": str(path)})
            meta = root / "meta.json"
            meta.write_text(json.dumps({"scanner": {
                "nVoxel": list(shape), "sVoxel": [4, 12, 24], "offOrigin": [0, 0, 0]}}))
            captured = []
            original = Axes.imshow

            def capture(axis, values, *args, **kwargs):
                captured.append((np.array(values), kwargs))
                return original(axis, values, *args, **kwargs)

            with mock.patch.object(Axes, "imshow", new=capture):
                report = images.render_grid(cases, meta, root / "images")
            self.assertEqual(len(captured), 18)
            for case_index in range(2):
                with np.load(root / "images/slices" / f"case{case_index}.npz") as slices:
                    self.assertEqual(len(slices.files), 9)
                    for normal, direction in enumerate("xyz"):
                        for column, fraction in enumerate((0.25, 0.5, 0.75)):
                            index = int(shape[normal] * fraction)
                            plane = np.take(base * (case_index + 1), index, axis=normal)
                            np.testing.assert_array_equal(slices[f"{direction}_{int(fraction * 100)}"], plane)
                            actual, kwargs = captured[case_index * 9 + normal * 3 + column]
                            np.testing.assert_array_equal(actual, plane.T)
                            self.assertEqual(kwargs["origin"], "lower")
                            self.assertEqual([kwargs["vmin"], kwargs["vmax"]], report["shared_window"])
                self.assertTrue((root / "images" / f"case{case_index}.png").is_file())
            self.assertEqual(captured[0][1]["extent"], (-6, 6, -12, 12))
            self.assertEqual([images.slice_index(256, f) for f in images.FRACTIONS], [64, 128, 192])

    def test_mismatched_grids_and_nonfinite_volumes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "volume.npy"
            np.save(path, np.zeros((2, 3, 4)))
            with self.assertRaisesRegex(ValueError, "grid changed"):
                images.load_volume(path, (4, 3, 2))
            values = np.zeros((2, 3, 4))
            values[0, 0, 0] = np.nan
            np.save(path, values)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                images.load_volume(path)


if __name__ == "__main__":
    unittest.main()
