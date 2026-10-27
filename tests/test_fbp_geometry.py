"""Physical-coordinate regressions, independent of CUDA/TIGRE."""

import ast
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from fbp_test_support import import_prepare
import run_pipeline as pipeline
from r2_gaussian.utils.geometry_utils import (
    initialization_positions,
    parallel_detector_half_extents,
    parallel_projection_matrix,
    validate_physical_dataset,
)

prepare = import_prepare()
ROOT = Path(__file__).resolve().parents[1]


def scanner(h=204, w=336):
    return dict(
        mode="parallel",
        nDetector=[h, w],
        sDetector=[h * 0.00004, w * 0.00004],
        sVoxel=[max(h, w) * 0.00004] * 3,
        offOrigin=[0, 0, 0],
        offDetector=[-0.00037, 0.00008],
    )


def normalized(cfg):
    return {**cfg, "sDetector": (np.array(cfg["sDetector"]) * 2 / max(cfg["sVoxel"])).tolist()}


def camera_to_world(distance, angle, offset):
    tree = ast.parse((ROOT / "r2_gaussian/dataset/dataset_readers.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "angle2pose")
    namespace = dict(np=np)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "dataset_readers.py", "exec"), namespace)
    return namespace["angle2pose"](distance, angle, offset)


def graphics_functions():
    # Run production matrix builders with a NumPy facade for torch tensors.
    tree = ast.parse((ROOT / "r2_gaussian/utils/graphics_utils.py").read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("getProjectionMatrix", "getParallelProjectionMatrix")
    ]
    namespace = dict(
        torch=SimpleNamespace(
            float32=np.float32,
            as_tensor=lambda v, dtype=None, device=None: np.asarray(v, dtype=dtype),
        ),
        parallel_projection_matrix=parallel_projection_matrix,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "graphics_utils.py", "exec"), namespace)
    return namespace


class PhysicalProjectionTests(unittest.TestCase):
    def test_off_center_points_match_physical_pixels_across_angles(self):
        for h, w in [(204, 336), (400, 472), (428, 616), (608, 604), (256, 256)]:
            cfg = scanner(h, w)
            scale = 2 / max(cfg["sVoxel"])
            projection = parallel_projection_matrix(normalized(cfg))
            points = np.array([[0.001, -0.002, 0.0007], [-0.0013, 0.0011, -0.0005], [0, 0, 0]])
            for angle in [0, 0.4, 1.3, 2.1]:
                u = (
                    -points[:, 0] * np.sin(angle)
                    + points[:, 1] * np.cos(angle)
                    - cfg["offDetector"][0]
                )
                v = -points[:, 2] - cfg["offDetector"][1]
                view = np.linalg.inv(
                    camera_to_world(8, angle, np.array(cfg["offDetector"]) * scale)
                )
                world = np.column_stack([points * scale, np.ones(len(points))])
                camera = world @ view.T
                np.testing.assert_allclose(
                    camera[:, :2], np.column_stack([u, v]) * scale, atol=1e-12
                )
                ndc = (camera @ projection.T)[:, :2]
                pixels = (ndc + 1) * np.array([w, h]) / 2 - 0.5
                expected = np.column_stack([u, v]) / 0.00004 + (np.array([w, h]) - 1) / 2
                np.testing.assert_allclose(pixels, expected, atol=2e-5)

    def test_sphere_radius_and_integration_length(self):
        cfg = scanner()
        cfg["sDetector"][0] *= 1.3
        half = parallel_detector_half_extents(normalized(cfg))
        scale = 2 / max(cfg["sVoxel"])
        radius = 0.0002
        pixel_scales = np.array(cfg["nDetector"])[::-1] / (2 * np.array(half))
        expected = radius / (np.array(cfg["sDetector"])[::-1] / np.array(cfg["nDetector"])[::-1])
        np.testing.assert_allclose(radius * scale * pixel_scales, expected)
        covariance = np.diag([*(radius * scale * pixel_scales) ** 2, (radius * scale) ** 2])
        mu = np.sqrt(2 * np.pi * np.linalg.det(covariance) / np.linalg.det(covariance[:2, :2]))
        self.assertAlmostEqual(mu, np.sqrt(2 * np.pi) * radius * scale)

    def test_analytic_disk_fbp_recovers_physical_center_and_radius(self):
        # Independent CPU ramp filter/backprojection in normalized physical
        # units, with detector span deliberately different from volume size.
        count, output_size, detector_span = 512, 128, 1.7
        center, radius = np.array([0.19, -0.13]), 0.17
        angles = np.arange(180) * np.pi / 180
        pitch = detector_span / count
        detector = (np.arange(count) + 0.5) * pitch - detector_span / 2
        projected = center[0] * np.cos(angles) + center[1] * np.sin(angles)
        sino = 2 * np.sqrt(np.maximum(radius**2 - (detector[None, :] - projected[:, None]) ** 2, 0))
        padded = 2048
        ramp = np.abs(np.fft.rfftfreq(padded, d=pitch))
        filtered = np.fft.irfft(np.fft.rfft(sino, n=padded, axis=1) * ramp, n=padded, axis=1)[
            :, :count
        ]
        axis = (np.arange(output_size) + 0.5) * 2 / output_size - 1
        x, y = np.meshgrid(axis, axis, indexing="ij")
        image = np.zeros_like(x)
        for index, angle in enumerate(angles):
            ray = x * np.cos(angle) + y * np.sin(angle)
            image += np.interp(ray, detector, filtered[index], left=0, right=0)
        mask = image > image.max() * 0.5
        np.testing.assert_allclose([x[mask].mean(), y[mask].mean()], center, atol=0.008)
        self.assertAlmostEqual(
            np.sqrt(mask.sum() * (2 / output_size) ** 2 / np.pi), radius, delta=0.008
        )

    def test_production_base_and_pose_matrices_preserve_scale_and_pixel_shifts(self):
        functions = graphics_functions()
        cfg = normalized(scanner())
        base = functions["getProjectionMatrix"](0, 0, 0, cfg)
        zero = functions["getParallelProjectionMatrix"]([0, 0], cfg, device="cpu")
        np.testing.assert_array_equal(base, zero)
        h, w = cfg["nDetector"]
        shift_px = np.array([2.25, -1.5])
        shift_scene = shift_px * np.array(cfg["sDetector"])[::-1] / [w, h]
        shifted = functions["getParallelProjectionMatrix"](shift_scene, cfg, device="cpu")
        np.testing.assert_allclose((shifted - zero)[:2, 3] * [w / 2, h / 2], shift_px, atol=1e-6)
        np.testing.assert_array_equal(shifted[:2, :2], zero[:2, :2])

    def test_render_covariance_half_extents_for_both_backends(self):
        tree = ast.parse((ROOT / "r2_gaussian/gaussian/render_query.py").read_text())
        render = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
        branch = next(
            n for n in render.body if isinstance(n, ast.If) and ast.unparse(n.test) == "mode == 0"
        )
        for pose in [False, True]:
            namespace = dict(
                mode=0,
                pipe=SimpleNamespace(pose_enabled=pose),
                viewpoint_camera=SimpleNamespace(scanner_cfg=normalized(scanner())),
                pc=SimpleNamespace(get_xyz=SimpleNamespace(device="cpu")),
                torch=SimpleNamespace(
                    float32=np.float32,
                    tensor=lambda v, dtype=None, device=None: np.asarray(v, dtype=dtype),
                ),
                parallel_detector_half_extents=parallel_detector_half_extents,
            )
            exec(
                compile(ast.Module(body=[branch], type_ignores=[]), "render_query.py", "exec"),
                namespace,
            )
            np.testing.assert_allclose(
                [float(np.asarray(namespace[k]).reshape(-1)[0]) for k in ["tanfovx", "tanfovy"]],
                [1, 204 / 336],
            )

    def test_invalid_detector_extents(self):
        for size in [[0, 2], [1, float("nan")], [1, 2, 3]]:
            with self.assertRaises(ValueError):
                parallel_projection_matrix({"sDetector": size})

    def test_non_cubic_offset_voxel_positions(self):
        cfg = dict(sVoxel=[4, 2, 1], offOrigin=[1, -2, 0.5])
        result = initialization_positions([[0, 0, 0], [7, 3, 1]], [8, 4, 2], cfg)
        np.testing.assert_allclose(result, [[-0.375, -1.375, 0.125], [1.375, -0.625, 0.375]])

    def test_centered_cube_positions_preserved(self):
        indices = np.array([[0, 1, 2], [15, 12, 9]])
        np.testing.assert_allclose(
            initialization_positions(indices, [16] * 3, scanner()),
            initialization_positions(indices, [16] * 3),
            atol=1e-7,
        )

    def test_virtual_dataset_rejected(self):
        validate_physical_dataset({"scanner": scanner()})
        with self.assertRaisesRegex(ValueError, "removed"):
            validate_physical_dataset({"reconstruction": {"coordinate_mode": "gaussian"}})

    def test_removed_cli_and_yaml_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.build_parser().parse_args(["--fbp_coordinate_mode", "gaussian"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare.build_parser().parse_args(["--coordinate_mode", "gaussian"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.yaml"
            path.write_text("fbp: {coordinate_mode: gaussian}")
            with self.assertRaisesRegex(ValueError, "Unknown keys"):
                pipeline.resolve_args(
                    pipeline.build_parser().parse_args(["--pipeline_config", str(path)])
                )

    def test_physical_fbp_rectangular_detector_and_offsets(self):
        args = prepare.build_parser().parse_args(
            [
                "--input_dir",
                "input",
                "--output_dir",
                "out",
                "--sVoxel",
                "0.01344",
                "0.01344",
                "0.01344",
                "--pixel_size",
                "0.00004",
                "--pixel_subsample",
                "1",
                "--offDetector",
                "-0.00037",
                "0.00008",
            ]
        )
        geo = prepare.build_geometry(args, [204, 336])
        np.testing.assert_allclose(geo.sDetector, [0.00816, 0.01344])
        np.testing.assert_allclose(geo.dDetector, [0.00004, 0.00004])
        np.testing.assert_allclose(geo.offDetector, [-0.00037, 0.00008])


if __name__ == "__main__":
    unittest.main()
