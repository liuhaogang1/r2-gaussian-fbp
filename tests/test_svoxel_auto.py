"""Automatic FOV sizing and integration tests without TIGRE/CUDA."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import tifffile

from fbp_test_support import import_prepare
prepare = import_prepare()
import run_pipeline as pipeline
from scripts.auto_svoxel import (
    auto_svoxel_from_projection, detector_shape_after_preprocessing, parse_svoxel,
)
from scripts.fbp_preprocess import process_projection


class AutoSizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()

    def projection(self, shape, name="Tomo_001.tiff"):
        path = self.source / name
        tifffile.imwrite(path, np.ones(shape, dtype=np.float32))
        return path

    def pipeline_args(self, *extra):
        return pipeline.resolve_args(pipeline.build_parser().parse_args([
            "--input_dir", str(self.source), "--output_dir", str(self.root / "out"),
            "--model_path", str(self.root / "model"), "--fbp_sVoxel", "auto",
            "--fbp_pixel_size", "0.00004", "--fbp_pixel_subsample", "1", *extra]))

    def test_each_dataset_uses_its_own_projection_header(self):
        for shape, length in [((204, 336), .01344), ((360, 476), .01904), ((608, 604), .02432)]:
            path = self.projection(shape)
            values, info = auto_svoxel_from_projection(path, .00004, 1)
            np.testing.assert_allclose(values, [length] * 3)
            self.assertEqual(info["raw_shape"], list(shape))

    def test_shape_prediction_matches_real_preprocessing(self):
        args = prepare.build_parser().parse_args(["--input_dir", "in", "--output_dir", "out",
            "--input_type", "line_integral", "--shift_v", "0"])
        for shape in ((360, 476), (608, 604), (15, 29), (29, 15), (1, 1)):
            for factor in (1, 2, 4):
                args.pixel_subsample = factor
                actual = process_projection(np.ones(shape, dtype=np.float32), args)
                self.assertEqual(list(actual.shape), detector_shape_after_preprocessing(shape, factor))

    def test_downsampling_crop_and_rounding_are_included(self):
        path = self.projection((360, 476))
        values, info = auto_svoxel_from_projection(path, .00004, 4)
        self.assertEqual(info["processed_shape"], [90, 91])
        np.testing.assert_allclose(values, [91 * .00004 * 4] * 3)

    def test_invalid_manual_and_auto_values_are_rejected(self):
        self.assertEqual(parse_svoxel("AUTO"), "auto")
        self.assertEqual(parse_svoxel([1, 2, 3]), [1., 2., 3.])
        for bad in ([], [1], [1, 2], [1, 2, 3, 4], [1, 0, 3],
                    [1, float("nan"), 3], [1, float("inf"), 3], ["auto", 1, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_svoxel(bad)
        path = self.projection((6, 10))
        for pitch in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                auto_svoxel_from_projection(path, pitch, 1)
        for shape in ((4, 5, 3), (0, 5)):
            with self.assertRaises(ValueError):
                detector_shape_after_preprocessing(shape, 1)

    def test_both_parsers_accept_auto_and_manual_but_reject_partial(self):
        for parser, base, option, attribute in (
            (prepare.build_parser(), ["--input_dir", "in", "--output_dir", "out"], "--sVoxel", "sVoxel"),
            (pipeline.build_parser(), [], "--fbp_sVoxel", "fbp_sVoxel"),
            (pipeline.build_parser(), [], "--sVoxel", "fbp_sVoxel"),
        ):
            self.assertEqual(getattr(parser.parse_args([*base, option, "auto"]), attribute), "auto")
            self.assertEqual(getattr(parser.parse_args([*base, option, "1", "2", "3"]), attribute), [1., 2., 3.])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args([*base, option, "1", "2"])

    def test_pipeline_resolves_current_input_and_keeps_auto_for_prepare(self):
        self.projection((360, 476))
        args = self.pipeline_args("--fbp_recon_pixel_subsample", "4")
        pipeline.validate_args(args)
        pipeline.resolve_pipeline_svoxel(args)
        np.testing.assert_allclose(args.fbp_sVoxel, [.01904] * 3)
        self.assertEqual(args.fbp_sVoxel_requested, "auto")
        command = pipeline.build_fbp_command(args, None, [0, 0])
        self.assertEqual(command[command.index("--sVoxel") + 1], "auto")
        self.assertEqual(args.fbp_sVoxel_resolution["processed_shape"], [360, 476])

    def test_continuing_train_uses_existing_grid_without_raw_tiffs(self):
        args = self.pipeline_args("--start_from", "train")
        args.output_dir.mkdir()
        (args.output_dir / "meta_data.json").write_text(json.dumps({"scanner": {"sVoxel": [.01, .02, .03]}}))
        pipeline.resolve_pipeline_svoxel(args)
        self.assertEqual(args.fbp_sVoxel, [.01, .02, .03])
        self.assertEqual(args.fbp_sVoxel_resolution["method"], "existing_dataset")

    def test_dry_run_manifest_has_resolved_numbers_and_auto_provenance(self):
        self.projection((360, 476))
        args = self.pipeline_args("--dry_run", "--stop_after", "fbp", "--skip_center")
        pipeline.main(args)
        manifest = json.loads((args.output_dir / "pipeline_manifest.json").read_text())
        resolved = manifest["resolved_arguments"]
        np.testing.assert_allclose(resolved["fbp_sVoxel"], [.01904] * 3)
        self.assertEqual(resolved["fbp_sVoxel_requested"], "auto")
        self.assertEqual(resolved["fbp_sVoxel_resolution"]["processed_shape"], [360, 476])

    def test_yaml_auto_and_explicit_values(self):
        path = self.root / "config.yaml"
        for value, expected in (("auto", "auto"), ("[1, 2, 3]", [1., 2., 3.])):
            path.write_text(f"fbp:\n  sVoxel: {value}\n")
            args = pipeline.resolve_args(pipeline.build_parser().parse_args(["--pipeline_config", str(path)]))
            self.assertEqual(args.fbp_sVoxel, expected)

    def test_prepare_metadata_and_volume_match_manual_auto_result(self):
        for i in range(3):
            self.projection((8, 12), f"Tomo_{i}.tiff")
        metadata = []
        for name, size in (("auto", ["auto"]), ("manual", ["0.00048"] * 3)):
            args = prepare.build_parser().parse_args(["--input_dir", str(self.source),
                "--output_dir", str(self.root / name), "--sVoxel", *size,
                "--pixel_size", "0.00004", "--input_type", "line_integral", "--shift_v", "0",
                "--n_train", "2", "--n_test", "1"])
            # Geometry-dependent CPU stand-in checks the real stage wiring.
            with mock.patch.object(prepare, "reconstruct_with_optional_subsample",
                side_effect=lambda a, *x, **kw: (np.full((4, 4, 4), max(a.sVoxel), dtype=np.float32), 1, False)
            ), mock.patch.object(prepare, "save_fbp_preview", return_value=(0, 1)):
                prepare.main(args)
            metadata.append(json.loads((self.root / name / "meta_data.json").read_text()))
        np.testing.assert_allclose(metadata[0]["scanner"]["sVoxel"], [.00048] * 3)
        self.assertEqual(metadata[0]["source"]["sVoxel_resolution"]["requested"], "auto")
        for key, value in metadata[0]["scanner"].items():
            if key == "sVoxel":
                np.testing.assert_allclose(value, metadata[1]["scanner"][key], rtol=1e-12)
            else:
                self.assertEqual(value, metadata[1]["scanner"][key])
        np.testing.assert_array_equal(np.load(self.root / "auto" / "vol_fbp.npy"),
                                      np.load(self.root / "manual" / "vol_fbp.npy"))

    def test_flat_field_default_glob_excludes_flat_header(self):
        self.projection((2, 2), "Flat_001.tiff")
        self.projection((360, 476))
        args = self.pipeline_args("--fbp_flat_field")
        pipeline.resolve_pipeline_svoxel(args)
        np.testing.assert_allclose(args.fbp_sVoxel, [.01904] * 3)


if __name__ == "__main__":
    unittest.main()
