"""Coordinate regression tests, independent of CUDA/TIGRE installation."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import init_from_fbp as init
from fbp_test_support import import_prepare
prepare = import_prepare()
import run_pipeline as pipeline
from scripts.fbp_geometry import initialization_positions, reconstruction_scanner


def scanner():
    return dict(mode="parallel", sVoxel=[0.01344] * 3, nVoxel=[24] * 3,
                sDetector=[0.02432, 0.02416], nDetector=[608, 604],
                offOrigin=[0, 0, 0], offDetector=[0.00012, -0.00004],
                DSD=7, DSO=5, accuracy=0.5, filter="hann")


class GeometryTests(unittest.TestCase):
    def test_rectangular_detector_matches_gaussian_pixel_centers(self):
        original = scanner()
        adapted = reconstruction_scanner(original, "gaussian")
        length = max(original["sVoxel"])
        for angle in (0, 0.4, 1.3, 2.1):
            # Independent existing-renderer equation, including fixed offset.
            x, y, z = 0.002, -0.001, 0.003
            camera_u = -x * np.sin(angle) + y * np.cos(angle) - original["offDetector"][0]
            camera_v = z - original["offDetector"][1]
            gaussian = np.array([(camera_v * 2 / length + 1) * 608 / 2 - 0.5,
                                 (camera_u * 2 / length + 1) * 604 / 2 - 0.5])
            analytical = np.array([camera_v, camera_u]) / (
                np.array(adapted["sDetector"]) / adapted["nDetector"])
            analytical += (np.array(adapted["nDetector"]) - 1) / 2
            np.testing.assert_allclose(analytical, gaussian, atol=1e-10)
        self.assertEqual(original, scanner())
        self.assertEqual(adapted["offDetector"], original["offDetector"])

    def test_known_sphere_center_and_radius_in_detector_coordinates(self):
        cfg = reconstruction_scanner(scanner(), "gaussian")
        scale = 2 / max(cfg["sVoxel"])
        center, radius = np.array([0.24, -0.31, 0.18]), 0.09
        for count, axis in ((604, 1), (608, 2)):
            physical_center = center[axis] / scale
            physical_radius = radius / scale
            detector_pitch = max(cfg["sVoxel"]) / count
            self.assertAlmostEqual(physical_center / detector_pitch, center[axis] * count / 2)
            self.assertAlmostEqual(physical_radius / detector_pitch, radius * count / 2)

    def test_physical_geometry_is_preserved(self):
        cfg = scanner()
        self.assertEqual(reconstruction_scanner(cfg), cfg)
        self.assertIsNot(reconstruction_scanner(cfg), cfg)
        cfg["mode"] = "cone"
        self.assertEqual(reconstruction_scanner(cfg), cfg)
        with self.assertRaisesRegex(ValueError, "parallel"):
            reconstruction_scanner(cfg, "gaussian")

    def test_analytic_disk_fbp_recovers_gaussian_center_and_radius(self):
        # Independent CPU ramp-filter/backprojection of analytic line integrals.
        # The disk sinogram uses the existing Gaussian normalized detector.
        count, output_size = 604, 128
        center, radius = np.array([0.22, -0.14]), 0.16
        angles = np.arange(180) * np.pi / 180
        t = (np.arange(count) + 0.5) * 2 / count - 1
        projected_center = center[0] * np.cos(angles) + center[1] * np.sin(angles)
        sino = 2 * np.sqrt(np.maximum(radius ** 2 - (t[None, :] - projected_center[:, None]) ** 2, 0))
        padded = 2048
        ramp = np.abs(np.fft.rfftfreq(padded, d=2 / count))
        filtered = np.fft.irfft(np.fft.rfft(sino, n=padded, axis=1) * ramp, n=padded, axis=1)[:, :count]
        axis = (np.arange(output_size) + 0.5) * 2 / output_size - 1
        x, y = np.meshgrid(axis, axis, indexing="ij")
        def reconstruct(detector_span):
            result = np.zeros_like(x)
            for i, angle in enumerate(angles):
                ray = (x * np.cos(angle) + y * np.sin(angle)) * 2 / detector_span
                result += np.interp(ray, t, filtered[i], left=0, right=0)
            mask = result > result.max() * 0.5
            return np.array([x[mask].mean(), y[mask].mean()]), np.sqrt(mask.sum() * (2 / output_size) ** 2 / np.pi)
        cfg = scanner()
        length = max(cfg["sVoxel"])
        virtual = reconstruction_scanner(cfg, "gaussian")
        actual_center, actual_radius = reconstruct(virtual["sDetector"][1] * 2 / length)
        np.testing.assert_allclose(actual_center, center, atol=0.008)
        self.assertAlmostEqual(actual_radius, radius, delta=0.008)
        old_center, old_radius = reconstruct(cfg["sDetector"][1] * 2 / length)
        ratio = cfg["sDetector"][1] / length
        np.testing.assert_allclose(old_center, center * ratio, atol=0.008)
        self.assertAlmostEqual(old_radius, radius * ratio, delta=0.008)

    def test_non_cubic_offset_voxel_positions(self):
        cfg = scanner()
        cfg.update(sVoxel=[4, 2, 1], offOrigin=[1, -2, 0.5])
        indices = np.array([[0, 0, 0], [7, 3, 1]])
        result = initialization_positions(indices, [8, 4, 2], cfg)
        np.testing.assert_allclose(result, [[-0.375, -1.375, 0.125], [1.375, -0.625, 0.375]])

    def test_centered_cube_point_mapping_is_unchanged(self):
        indices = np.array([[0, 1, 2], [15, 12, 9]])
        np.testing.assert_allclose(initialization_positions(indices, [16] * 3, scanner()),
                                   initialization_positions(indices, [16] * 3), atol=1e-7)

    def test_gaussian_previews_use_same_spacing_as_test_output(self):
        args = SimpleNamespace(sVoxel=[0.01344] * 3, coordinate_mode="gaussian")
        np.testing.assert_allclose(prepare.preview_size(args) / 256, [0.0078125] * 3)
        args.coordinate_mode = "physical"
        np.testing.assert_allclose(prepare.preview_size(args), [0.01344] * 3)

    def test_full_fbp_geometry_and_recon_only_downsampling(self):
        args = prepare.build_parser().parse_args(["--input_dir", "input", "--output_dir", "out",
            "--coordinate_mode", "gaussian", "--sVoxel", "0.01344", "0.01344", "0.01344",
            "--pixel_size", "0.00004", "--pixel_subsample", "1", "--recon_pixel_subsample", "2"])
        fake_tigre = ModuleType("tigre")
        fake_tigre.geometry = lambda **kwargs: SimpleNamespace(**kwargs)
        projections = np.arange(3 * 12 * 20, dtype=np.float32).reshape(3, 12, 20)
        with mock.patch.object(prepare, "tigre", fake_tigre):
            with mock.patch.object(prepare, "reconstruct_volume", return_value=np.ones((4, 4, 4))) as reconstruct:
                prepare.reconstruct_with_optional_subsample(args, projections, np.zeros(3), None)
        saved_input, _, _, geo = reconstruct.call_args.args
        self.assertEqual(saved_input.shape, (3, 6, 10))
        np.testing.assert_allclose(geo.sDetector, [0.01344, 0.01344])
        self.assertEqual(projections.shape, (3, 12, 20))
        np.testing.assert_allclose(saved_input[:, 0, 0], projections[:, 0, 0])
        np.testing.assert_allclose(saved_input[:, -1, -1], projections[:, -1, -1])

    def test_pipeline_forwards_mode_to_full_and_train_fbp(self):
        args = pipeline.resolve_args(pipeline.build_parser().parse_args(["--input_dir", "input",
            "--output_dir", "out", "--model_path", "model", "--fbp_coordinate_mode", "gaussian",
            "--init_source", "train_fbp", "--train_slice_loss_enabled"]))
        for command in (pipeline.build_fbp_command(args, None, [0, 0]),
                        pipeline.build_init_command(args, Path("init.npy"))):
            self.assertEqual(command[command.index("--coordinate_mode") + 1], "gaussian")
        self.assertEqual(args.train_slice_reference_path.name, "vol_train_fbp_gaussian.npy")

    def test_dataset_writer_preserves_training_data_and_records_fbp_mode(self):
        import tifffile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.mkdir()
            for i in range(3):
                tifffile.imwrite(source / f"view_{i}.tiff", np.ones((8, 12), dtype=np.float32) * (i + 1))
            fake_tigre = ModuleType("tigre")
            fake_tigre.geometry = lambda **kwargs: SimpleNamespace(**kwargs)
            metadata = []
            for mode in ("physical", "gaussian"):
                args = prepare.build_parser().parse_args(["--input_dir", str(source),
                    "--output_dir", str(root / mode), "--coordinate_mode", mode,
                    "--input_type", "line_integral", "--shift_v", "0", "--n_train", "2",
                    "--n_test", "1", "--nVoxel", "8", "8", "8",
                    "--sVoxel", "0.01344", "0.01344", "0.01344", "--pixel_size", "0.00004"])
                with mock.patch.object(prepare, "tigre", fake_tigre), mock.patch.object(
                    prepare, "reconstruct_volume", return_value=np.ones((8, 8, 8), dtype=np.float32)
                ), mock.patch.object(prepare, "save_fbp_preview", return_value=(0, 1)) as preview:
                    prepare.main(args)
                expected_size = [2, 2, 2] if mode == "gaussian" else [0.01344] * 3
                np.testing.assert_allclose(preview.call_args.args[2], expected_size)
                metadata.append(json.loads((root / mode / "meta_data.json").read_text()))
                if mode == "gaussian":
                    self.assertEqual(metadata[-1]["reconstruction"]["coordinate_mode"], mode)
                else:
                    self.assertNotIn("reconstruction", metadata[-1])
            self.assertEqual(metadata[0]["scanner"], metadata[1]["scanner"])
            self.assertEqual(metadata[0]["proj_train"], metadata[1]["proj_train"])
            np.testing.assert_array_equal(np.load(root / "physical" / "proj_all.npy"),
                                          np.load(root / "gaussian" / "proj_all.npy"))


class TrainFBPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dataset = Path(self.temp.name)
        self.projection = self.dataset / "view.npy"
        np.save(self.projection, np.ones((6, 10), dtype=np.float32))
        self.metadata = dict(scanner=scanner(), proj_train=[dict(file_path="view.npy", angle=0)],
                             reconstruction=dict(coordinate_mode="gaussian"))
        self.save_metadata()
        self.args = argparse.Namespace(dataset=self.dataset, use_train_fbp=True, volume=None,
                                       fbp_output=None, coordinate_mode="gaussian")

    def save_metadata(self):
        (self.dataset / "meta_data.json").write_text(json.dumps(self.metadata))

    def test_train_view_path_uses_same_virtual_geometry(self):
        backend = ModuleType("r2_gaussian.utils.ct_utils")
        backend.get_geometry_tigre = mock.Mock(side_effect=lambda cfg: cfg)
        backend.recon_volume = mock.Mock(return_value=np.ones((24, 24, 24)))
        with mock.patch.dict(sys.modules, {"r2_gaussian.utils.ct_utils": backend}):
            init.reconstruct_from_train_views(self.dataset, "gaussian")
        geometry = backend.get_geometry_tigre.call_args.args[0]
        self.assertEqual(geometry["sDetector"], [0.01344, 0.01344])
        self.assertEqual(self.metadata["scanner"]["sDetector"], [0.02432, 0.02416])

    def test_cached_geometry_and_changed_angles_are_checked(self):
        with mock.patch.object(init, "reconstruct_from_train_views", return_value=np.ones((24, 24, 24))) as reconstruct:
            _, output = init.load_initial_volume(self.args)
            self.assertEqual(output.name, "vol_train_fbp_gaussian.npy")
            init.load_initial_volume(self.args)
            self.assertEqual(reconstruct.call_count, 1)
            self.metadata["scanner"]["offDetector"][0] += 0.0001
            self.save_metadata()
            init.load_initial_volume(self.args)
            self.assertEqual(reconstruct.call_count, 2)
            self.metadata["proj_train"][0]["angle"] = 0.2
            self.save_metadata()
            init.load_initial_volume(self.args)
            self.assertEqual(reconstruct.call_count, 3)
            np.save(self.projection, np.ones((7, 10), dtype=np.float32))
            init.load_initial_volume(self.args)
            self.assertEqual(reconstruct.call_count, 4)

    def test_unverified_custom_cache_is_not_reused(self):
        self.args.fbp_output = self.dataset / "custom.npy"
        np.save(self.args.fbp_output, np.zeros((24, 24, 24)))
        with mock.patch.object(init, "reconstruct_from_train_views", return_value=np.ones((24, 24, 24))) as reconstruct:
            volume, _ = init.load_initial_volume(self.args)
        reconstruct.assert_called_once()
        self.assertTrue(np.all(volume == 1))


if __name__ == "__main__":
    unittest.main()
