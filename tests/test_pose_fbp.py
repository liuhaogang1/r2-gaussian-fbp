"""Data binding and actual offline CLI flows, without training dependencies."""

import contextlib
import gzip
import io
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts import apply_pose_fbp as cli
from r2_gaussian.utils.artifact_utils import dataset_identity, file_identity
from r2_gaussian.utils.pose_fbp import (correct_projection, export_projections,
                                       load_inputs, read_exported_projections)


class PoseFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "proj_train").mkdir()
        self.pose_path = self.root / "iteration_12" / "pose_offsets.npz"
        self.pose_path.parent.mkdir()
        self.scanner = dict(mode="parallel", DSO=5.0, DSD=10.0, nDetector=[18, 22],
                            sDetector=[1.8, 2.2], nVoxel=[7, 9, 5], sVoxel=[1.0, 1.2, 0.8],
                            offOrigin=[0.1, -0.1, 0.02], offDetector=[0.03, -0.02],
                            accuracy=0.5, filter="hann")
        self.angles = np.array([0.1, 0.6, 1.4])
        self.metadata = {"scanner": self.scanner, "proj_train": []}
        vv, uu = np.indices(self.scanner["nDetector"])
        for index, angle in enumerate(self.angles):
            name = f"proj_train/view{index}.float.npy"
            projection = ((uu - 8) * 0.02 + (vv - 9) * 0.03 + index * 0.01).astype(np.float32)
            np.save(self.dataset / name, projection)
            self.metadata["proj_train"].append({"file_path": name, "angle": float(angle)})
        self.save_metadata()
        order = np.array([2, 0, 1])
        self.pose = dict(binding_version=np.array(1, np.int32), pose_format_version=np.array(2, np.int32),
                         model_iteration=np.array(12, np.int64), scanner_mode=np.array("parallel"),
                         dataset_identity=np.array(dataset_identity(self.dataset)),
                         image_ids=np.array(["view2", "view0", "view1"]), base_angles=self.angles[order],
                         delta_angle=np.array([-0.01, 0.004, 0.006])[order],
                         view_shift=np.array([[0.2, -0.1], [-0.1, 0.2], [-0.1, -0.1]])[order],
                         global_shift=np.array([0.3, -0.2]), detector_roll=np.array([0.008]),
                         beam_tilt=np.array([0.003, -0.002]))
        self.save_pose()

    def save_metadata(self):
        (self.dataset / "meta_data.json").write_text(json.dumps(self.metadata), encoding="utf-8")

    def save_pose(self):
        np.savez(self.pose_path, **self.pose)

    def inputs(self):
        return load_inputs(self.dataset, self.pose_path, 12)

    def args(self, output, *extra):
        return cli.build_parser().parse_args([
            "--dataset", str(self.dataset), "--pose", str(self.pose_path), "--iteration", "12",
            "--output-dir", str(self.root / output), "--backend", "cpu", *extra])


class PoseBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = PoseFixture(self.directory.name)

    def test_shuffled_pose_and_reordered_metadata_bind_by_main_image_id(self):
        fixture = self.fixture
        inputs = fixture.inputs()
        np.testing.assert_allclose(inputs.pose["delta_angle"], [-0.01, 0.004, 0.006])
        np.testing.assert_allclose(inputs.pose["view_shift"][0], [0.2, -0.1])
        self.assertEqual(inputs.image_ids, ["view0", "view1", "view2"])
        fixture.metadata["proj_train"].reverse()
        fixture.save_metadata()
        reversed_inputs = fixture.inputs()
        self.assertEqual(reversed_inputs.image_ids, inputs.image_ids[::-1])
        np.testing.assert_array_equal(reversed_inputs.pose["view_shift"], inputs.pose["view_shift"][::-1])
        self.assertEqual(reversed_inputs.dataset_hash, inputs.dataset_hash)

    def test_binding_mapping_agrees_with_the_actual_main_pipeline_validator(self):
        # Load the lightweight production file directly: the pose package's
        # __init__ imports torch even though artifacts.py itself needs no torch.
        path = Path(__file__).resolve().parents[1] / "r2_gaussian/pose/artifacts.py"
        spec = importlib.util.spec_from_file_location("offline_binding_reference", path)
        artifacts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(artifacts)
        fixture = self.fixture
        inputs = fixture.inputs()
        scene = SimpleNamespace(source_path=str(fixture.dataset), scanner_cfg=inputs.scanner,
                                train_cameras=[SimpleNamespace(image_name=name, angle=angle)
                                               for name, angle in zip(inputs.image_ids, inputs.angles)])
        indices = artifacts.validate_pose_file(fixture.pose_path, scene, 12)
        for name in ("delta_angle", "view_shift"):
            np.testing.assert_array_equal(inputs.pose[name], fixture.pose[name][indices])

    def test_invalid_binding_arrays_and_versions_are_rejected(self):
        fixture = self.fixture
        original = dict(fixture.pose)
        changes = [dict(model_iteration=np.array(11)), dict(binding_version=np.array(2)),
                   dict(pose_format_version=np.array(1)), dict(scanner_mode=np.array("cone")),
                   dict(dataset_identity=np.array("wrong")), dict(image_ids=np.array(["view0"] * 3)),
                   dict(base_angles=np.zeros(3)), dict(delta_angle=np.zeros((3, 1))),
                   dict(view_shift=np.zeros((2, 2))), dict(global_shift=np.zeros(3)),
                   dict(beam_tilt=np.array([0, np.nan])), dict(detector_roll=np.array([np.inf]))]
        for change in changes:
            with self.subTest(change=list(change)):
                fixture.pose = {**original, **change}
                fixture.save_pose()
                with self.assertRaises(ValueError):
                    fixture.inputs()
        fixture.pose = dict(original)
        del fixture.pose["binding_version"]
        fixture.save_pose()
        with self.assertRaisesRegex(ValueError, "binding_version"):
            fixture.inputs()

    def test_projection_content_is_bound_and_dimensions_validated(self):
        fixture = self.fixture
        source = fixture.dataset / fixture.metadata["proj_train"][0]["file_path"]
        original = np.load(source)
        np.save(source, original + 1)
        with self.assertRaisesRegex(ValueError, "dataset_identity"):
            fixture.inputs()
        np.save(source, np.zeros((17, 22), np.float32))
        with self.assertRaisesRegex(ValueError, "shape"):
            fixture.inputs()
        np.save(source, np.full(original.shape, np.nan, np.float32))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            fixture.inputs()

    def test_duplicate_main_ids_and_inconsistent_pitch_rejected(self):
        fixture = self.fixture
        fixture.metadata["proj_train"][1]["file_path"] = fixture.metadata["proj_train"][0]["file_path"]
        fixture.save_metadata()
        with self.assertRaisesRegex(ValueError, "unique"):
            fixture.inputs()
        fixture.scanner["dDetector"] = [0.2, 0.2]
        fixture.save_metadata()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            fixture.inputs()

    def test_disabled_pose_arrays_default_to_zero_but_nonzero_tilt_is_accepted(self):
        fixture = self.fixture
        for name in ("delta_angle", "view_shift", "global_shift", "detector_roll"):
            del fixture.pose[name]
        fixture.save_pose()
        inputs = fixture.inputs()
        self.assertTrue(np.any(inputs.pose["beam_tilt"] != 0))
        self.assertFalse(np.any(inputs.pose["view_shift"]))

    def test_zero_resampling_and_export_keep_values_and_order(self):
        fixture = self.fixture
        for name in ("delta_angle", "view_shift", "global_shift", "detector_roll"):
            fixture.pose[name] = np.zeros_like(fixture.pose[name])
        fixture.save_pose()
        inputs = fixture.inputs()
        output = fixture.root / "export"
        export_projections(inputs, output)
        stack, angles, manifest = read_exported_projections(output)
        np.testing.assert_array_equal(angles, inputs.angles)
        for index, record in enumerate(inputs.records):
            np.testing.assert_array_equal(stack[index], np.load(inputs.dataset / record["file_path"]))
            mask = np.load(output / manifest["views"][index]["mask"])
            self.assertEqual(mask.dtype, bool)
            self.assertTrue(mask.all())
        self.assertLess(float(stack.min()), 0)  # no display normalization or clipping

    def test_known_integer_shift_sign_and_missing_support(self):
        image = np.zeros((11, 17), np.float32)
        image[5, 9] = -3
        corrected, mask = correct_projection(image, [0.2, 0.1], [2, -1], 0)
        self.assertEqual(corrected[6, 7], -3)
        self.assertFalse(mask[0].any())
        self.assertFalse(mask[:, -2:].any())

    def test_saved_angle_manifest_mismatch_is_rejected(self):
        fixture = self.fixture
        output = fixture.root / "export"
        export_projections(fixture.inputs(), output)
        np.save(output / "angles_corrected.npy", np.zeros(3))
        with self.assertRaisesRegex(ValueError, "changed"):
            read_exported_projections(output)

    def test_saved_mask_is_bound_to_its_projection_record(self):
        fixture = self.fixture
        output = fixture.root / "export"
        export_projections(fixture.inputs(), output)
        np.save(output / "valid_masks/000000.npy", np.ones((2, 3), dtype=bool))
        with self.assertRaisesRegex(ValueError, "mask"):
            read_exported_projections(output)


class PoseCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = PoseFixture(self.directory.name)

    def run_quiet(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.run(args)

    def test_default_partial_ignores_nonzero_tilt_and_writes_physical_nifti(self):
        fixture = self.fixture
        before = dataset_identity(fixture.dataset)
        args = fixture.args("partial")
        report = self.run_quiet(args)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["ignored_parameters"], ["beam_tilt"])
        self.assertEqual(set(report["volumes"]), {"fbp_original", "fbp_partial"})
        self.assertEqual(dataset_identity(fixture.dataset), before)
        output = args.output_dir
        saved = np.load(output / "fbp_partial.npy")
        with gzip.open(output / "fbp_partial.nii.gz", "rb") as stream:
            data = stream.read()
        self.assertEqual(struct.unpack_from("<3h", data, 42), tuple(saved.shape))
        exported = np.frombuffer(data, dtype="<f4", offset=352).reshape(saved.shape, order="F")
        np.testing.assert_array_equal(exported, saved)
        spacing = np.asarray(fixture.scanner["sVoxel"]) / fixture.scanner["nVoxel"]
        np.testing.assert_allclose(struct.unpack_from("<3f", data, 80), spacing)
        self.assertTrue((output / "comparison_xyz.png").is_file())
        self.assertEqual(report["blink_comparison"]["status"], "written")
        self.assertEqual(report["blink_comparison"]["reference"]["selection"], "same_run_original")
        for normal in "xyz":
            self.assertTrue((output / f"blink_center_{normal}.gif").is_file())

    def test_ablation_exports_four_comparable_reconstructions(self):
        fixture = self.fixture
        args = fixture.args("ablation", "--ablation")
        report = self.run_quiet(args)
        self.assertEqual(len(report["volumes"]), 4)
        for name in ("shift_only", "shift_roll"):
            _, angles, _ = read_exported_projections(args.output_dir / "ablation" / name)
            np.testing.assert_array_equal(angles, fixture.angles)
        self.assertEqual(report["beam_tilt_application"], "ignored")

    def test_full_mode_accepts_nonzero_tilt_and_only_reconstructs_full_pose(self):
        fixture = self.fixture
        args = fixture.args("full", "--apply-beam-tilt")
        report = self.run_quiet(args)
        self.assertEqual(report["beam_tilt_application"], "3d_geometry")
        self.assertEqual(set(report["volumes"]), {"fbp_full_pose"})
        self.assertFalse((args.output_dir / "fbp_original.npy").exists())
        self.assertFalse((args.output_dir / "ablation").exists())
        self.assertTrue((args.output_dir / "geometry_full_pose.npz").exists())
        self.assertEqual(report["blink_comparison"]["status"], "skipped")
        self.assertFalse((args.output_dir / "blink_center_z.gif").exists())

    def test_full_blinks_use_existing_bound_train_fbp_and_one_reconstruction(self):
        from PIL import Image

        fixture = self.fixture
        path = fixture.dataset / "vol_train_fbp.npy"
        shape = tuple(fixture.scanner["nVoxel"])
        original = np.linspace(-0.4, 2, np.prod(shape), dtype=np.float32).reshape(shape)
        np.save(path, original)
        before = file_identity(path)
        path.with_suffix(".cache.json").write_text(json.dumps({
            "signature": {"dataset": dataset_identity(fixture.dataset)}, "artifact": before}), encoding="utf-8")
        args = fixture.args("full_blinks", "--apply-beam-tilt")
        with mock.patch.object(cli, "reconstruct_full", wraps=cli.reconstruct_full) as full, \
             mock.patch.object(cli, "reconstruct_standard", side_effect=AssertionError("no extra baseline")):
            report = self.run_quiet(args)
        self.assertEqual(full.call_count, 1)
        self.assertEqual(set(report["volumes"]), {"fbp_full_pose"})
        blink = report["blink_comparison"]
        self.assertEqual(blink["reference"]["binding"], "verified_dataset_and_volume_hash")
        self.assertEqual(blink["reference"]["identity"], before)
        self.assertEqual(file_identity(path), before)
        self.assertEqual(blink["shared_display_window"], report["preview"]["shared_display_window"])
        expected_window = np.percentile(original[original > 0], [1, 99.5])
        np.testing.assert_allclose(blink["shared_display_window"], expected_window)
        corrected = np.load(args.output_dir / "fbp_full_pose.npy")
        low, high = expected_window
        for axis, normal in enumerate("xyz"):
            with Image.open(args.output_dir / blink["files"][normal]) as gif:
                self.assertEqual(gif.n_frames, 2)
                self.assertEqual(gif.info["loop"], 0)
                for frame_index, volume in enumerate((original, corrected)):
                    gif.seek(frame_index)
                    self.assertEqual(gif.info["duration"], 900)
                    plane = np.take(volume, shape[axis] // 2, axis=axis)
                    expected = np.round(np.clip((plane - low) / (high - low), 0, 1).T[::-1] * 255).astype(np.uint8)
                    expected = expected.repeat(2, axis=0).repeat(2, axis=1)
                    left = (gif.width - expected.shape[1]) // 2
                    actual = np.asarray(gif.convert("L"))[44:44 + expected.shape[0], left:left + expected.shape[1]]
                    np.testing.assert_array_equal(actual, expected)

    def test_reference_binding_rejects_wrong_dataset_or_modified_volume(self):
        fixture = self.fixture
        path = fixture.dataset / "vol_train_fbp.npy"
        np.save(path, np.ones(fixture.scanner["nVoxel"], np.float32))
        sidecar = path.with_suffix(".cache.json")
        for dataset_hash, volume_hash in (("wrong", file_identity(path)["sha256"]),
                                          (dataset_identity(fixture.dataset), "wrong")):
            sidecar.write_text(json.dumps({"signature": {"dataset": dataset_hash},
                                          "artifact": {"sha256": volume_hash}}), encoding="utf-8")
            reference, info = cli._load_comparison_reference(fixture.inputs(), None)
            self.assertIsNone(reference)
            self.assertEqual(info["status"], "skipped")
            self.assertIn("does not match", info["reason"])
            with self.assertRaisesRegex(ValueError, "--comparison-volume"):
                cli._load_comparison_reference(fixture.inputs(), path)

    def test_explicit_reference_checks_grid_and_marks_missing_provenance(self):
        fixture = self.fixture
        path = fixture.root / "custom_fbp.npy"
        np.save(path, np.ones((2, 3, 4), np.float32))
        with self.assertRaisesRegex(ValueError, "nVoxel grid"):
            cli._load_comparison_reference(fixture.inputs(), path)
        np.save(path, np.ones(fixture.scanner["nVoxel"], np.float32))
        reference, info = cli._load_comparison_reference(fixture.inputs(), path)
        try:
            self.assertEqual(info["binding"], "unverified_no_cache_sidecar")
            self.assertEqual(info["selection"], "explicit")
        finally:
            reference._mmap.close()
        args = fixture.args("invalid_comparison", "--comparison-volume", str(path))
        with self.assertRaisesRegex(ValueError, "requires --apply-beam-tilt"):
            self.run_quiet(args)
        self.assertFalse(args.output_dir.exists())

    def test_missing_train_reference_does_not_substitute_all_view_volume(self):
        fixture = self.fixture
        np.save(fixture.dataset / "vol_fbp.npy", np.ones(fixture.scanner["nVoxel"], np.float32))
        reference, info = cli._load_comparison_reference(fixture.inputs(), None)
        self.assertIsNone(reference)
        self.assertEqual(info["status"], "skipped")
        self.assertEqual(Path(info["path"]).name, "vol_train_fbp.npy")

    def test_mode_flags_are_mutually_exclusive_before_any_output(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.fixture.args("invalid", "--ablation", "--apply-beam-tilt")
        self.assertFalse((self.fixture.root / "invalid").exists())

    def test_export_only_full_does_not_import_or_require_reconstruction_backend(self):
        args = self.fixture.args("export", "--apply-beam-tilt", "--export-only", "--backend", "tigre")
        with mock.patch.object(cli, "require_backend", side_effect=AssertionError("must not call")):
            report = self.run_quiet(args)
        self.assertEqual(report["status"], "exported")
        self.assertFalse(report["reconstruction_performed"])
        self.assertTrue((args.output_dir / "geometry_full_pose.npz").exists())
        self.assertFalse((args.output_dir / "fbp_full_pose.npy").exists())

    def test_full_mode_exports_tiny_nonzero_tilt_without_collapsing_it(self):
        fixture = self.fixture
        fixture.pose["beam_tilt"] = np.array([3e-8, -7e-9])
        fixture.save_pose()
        args = fixture.args("tiny_tilt", "--apply-beam-tilt", "--export-only")
        report = self.run_quiet(args)
        self.assertEqual(report["status"], "exported")
        self.assertEqual(report["beam_tilt_application"], "3d_geometry")
        with np.load(args.output_dir / "geometry_full_pose.npz", allow_pickle=False) as geometry:
            np.testing.assert_array_equal(geometry["beam_tilt"], fixture.pose["beam_tilt"])
            self.assertGreater(float(np.min(np.abs(geometry["tigre_angles"][:, 1]))), 2.9e-8)

    def test_existing_or_nested_output_is_rejected_without_writes(self):
        fixture = self.fixture
        args = fixture.args("existing", "--export-only")
        args.output_dir.mkdir()
        marker = args.output_dir / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_quiet(args)
        self.assertEqual(marker.read_text(), "keep")
        args.output_dir = fixture.dataset / "new-output"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_quiet(args)
        self.assertFalse(args.output_dir.exists())

    def test_reconstruction_failure_records_failure_without_full_to_partial_fallback(self):
        args = self.fixture.args("failed", "--apply-beam-tilt")
        with mock.patch.object(cli, "reconstruct_full", side_effect=RuntimeError("backend failed")), \
             mock.patch.object(cli, "reconstruct_standard", side_effect=AssertionError("no fallback")):
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                self.run_quiet(args)
        report = json.loads((args.output_dir / "run_report.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("backend failed", report["error"])

    def test_zero_pose_full_partial_and_original_are_identical(self):
        fixture = self.fixture
        for name in ("delta_angle", "view_shift", "global_shift", "detector_roll", "beam_tilt"):
            fixture.pose[name] = np.zeros_like(fixture.pose[name])
        fixture.save_pose()
        partial, full = fixture.args("zero_partial"), fixture.args("zero_full", "--apply-beam-tilt")
        self.run_quiet(partial)
        self.run_quiet(full)
        original = np.load(partial.output_dir / "fbp_original.npy")
        np.testing.assert_array_equal(original, np.load(partial.output_dir / "fbp_partial.npy"))
        np.testing.assert_array_equal(original, np.load(full.output_dir / "fbp_full_pose.npy"))

    def test_fbp_uses_the_saved_pixels_without_applying_pose_again(self):
        fixture = self.fixture
        args = fixture.args("saved_input")
        original_export = cli.export_projections

        def export_with_known_saved_change(inputs, output, **kwargs):
            views = original_export(inputs, output, **kwargs)
            # Change the exported measurement before reconstruction reads it.
            for view in views:
                path = output / view["projection"]
                np.save(path, np.load(path) * 1.7)
            return views

        with mock.patch.object(cli, "export_projections", side_effect=export_with_known_saved_change):
            self.run_quiet(args)
        projections, angles, _ = read_exported_projections(args.output_dir)
        expected = cli.reconstruct_standard(projections, angles, fixture.inputs().scanner, backend="cpu")
        np.testing.assert_array_equal(np.load(args.output_dir / "fbp_partial.npy"), expected)


if __name__ == "__main__":
    unittest.main()
