"""CPU geometry regressions for the differentiable parallel-pose path."""

import ast
import csv
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def production_pose_functions():
    """Load the production Camera matrix methods with a NumPy torch facade."""
    path = ROOT / "r2_gaussian/dataset/cameras.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    camera = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Camera"
    )
    wanted = {
        "_rot_x",
        "_rot_y",
        "_rot_z",
        "_correct_parallel_c2w",
        "_rotation_matrix_to_quaternion",
        "_quaternion_to_rotation_matrix",
    }
    methods = [
        node for node in camera.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    torch = SimpleNamespace(
        cos=np.cos,
        sin=np.sin,
        sqrt=np.sqrt,
        clamp=lambda value, min=None, max=None: np.clip(
            value,
            -np.inf if min is None else min,
            np.inf if max is None else max,
        ),
        diagonal=np.diagonal,
        argmax=np.argmax,
        where=np.where,
        stack=np.stack,
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                normalize=lambda value, dim: value / np.linalg.norm(value, axis=dim)
            )
        ),
        transpose=np.swapaxes,
        eye=lambda size, device=None: np.eye(size, dtype=np.float64),
        linalg=SimpleNamespace(inv=np.linalg.inv),
    )
    namespace = {"torch": torch}
    for method in methods:
        method.decorator_list = []
    exec(
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def production_angle2pose():
    """Load the production scanner trajectory without importing torch."""
    path = ROOT / "r2_gaussian/dataset/dataset_readers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "angle2pose"
    )
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["angle2pose"]


def production_center_and_bound():
    """Load the optimizer gauge/bound projection with NumPy operations."""
    path = ROOT / "r2_gaussian/pose/parallel_pose.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pose_class = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ParallelPose"
    )
    method = next(
        node for node in pose_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_center_and_bound"
    )
    method.decorator_list = []

    def clamp(value, min=None, max=None):
        return np.clip(
            value,
            -np.inf if min is None else min,
            np.inf if max is None else max,
        )

    torch = SimpleNamespace(
        Tensor=object,
        mean=lambda value, dim, keepdim: np.mean(value, axis=dim, keepdims=keepdim),
        amax=lambda value, dim, keepdim: np.max(value, axis=dim, keepdims=keepdim),
        abs=np.abs,
        clamp=clamp,
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["_center_and_bound"]


def production_save_pose_result():
    path = ROOT / "train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "save_pose_result"
    )
    namespace = {
        "os": os,
        "osp": os.path,
        "csv": csv,
        "np": np,
        "PARALLEL_POSE_FORMAT_VERSION": 2,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["save_pose_result"]


class ParallelPoseGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pose_functions = production_pose_functions()
        cls.angle2pose = staticmethod(production_angle2pose())

    def corrected(self, base_c2w, values, pose_index=0):
        methods = self.pose_functions
        camera = SimpleNamespace(
            _base_world_view_transform=np.linalg.inv(base_c2w).T,
            pose_index=pose_index,
            _rot_x=methods["_rot_x"],
            _rot_y=methods["_rot_y"],
            _rot_z=methods["_rot_z"],
        )
        return methods["_correct_parallel_c2w"](camera, values, "cpu")

    @staticmethod
    def empty_values(**overrides):
        values = {
            "delta_angle": None,
            "beam_tilt": None,
            "detector_roll": None,
        }
        values.update(overrides)
        return values

    def test_delta_angle_matches_scanner_trajectory_and_moves_center(self):
        angle = np.deg2rad(37.0)
        delta = np.deg2rad(0.4)
        off_detector = [0.13, -0.07]
        base = self.angle2pose(5.0, angle, off_detector)

        corrected = self.corrected(
            base,
            self.empty_values(delta_angle=np.array([delta])),
        )
        expected = self.angle2pose(5.0, angle + delta, off_detector)

        np.testing.assert_allclose(corrected, expected, atol=1e-12)
        self.assertGreater(np.linalg.norm(corrected[:3, 3] - base[:3, 3]), 0.01)

    def test_detector_roll_changes_detector_axes_only(self):
        base = self.angle2pose(5.0, np.deg2rad(37.0), [0.13, -0.07])
        roll = np.deg2rad(0.5)
        corrected = self.corrected(
            base,
            self.empty_values(detector_roll=np.array([roll])),
        )

        np.testing.assert_allclose(corrected[:3, 3], base[:3, 3], atol=1e-12)
        np.testing.assert_allclose(corrected[:3, 2], base[:3, 2], atol=1e-12)
        self.assertAlmostEqual(
            np.rad2deg(np.arccos(np.clip(base[:3, 0] @ corrected[:3, 0], -1, 1))),
            0.5,
            places=8,
        )

    def test_beam_tilt_changes_ray_but_keeps_center_and_rotation_valid(self):
        base = self.angle2pose(5.0, np.deg2rad(37.0), [0.13, -0.07])
        tilt = np.deg2rad([0.3, -0.2])
        corrected = self.corrected(
            base,
            self.empty_values(beam_tilt=tilt),
        )

        np.testing.assert_allclose(corrected[:3, 3], base[:3, 3], atol=1e-12)
        self.assertGreater(
            np.linalg.norm(corrected[:3, 2] - base[:3, 2]),
            1e-4,
        )
        rotation = corrected[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)

    def test_global_only_pose_skips_per_view_values(self):
        base = self.angle2pose(5.0, np.deg2rad(37.0), [0.0, 0.0])
        corrected = self.corrected(
            base,
            self.empty_values(delta_angle=np.array([0.1]), detector_roll=np.array([0.2])),
            pose_index=None,
        )
        expected = base @ self.pose_functions["_rot_z"](0.2, "cpu")
        np.testing.assert_allclose(corrected, expected, atol=1e-12)

    def test_zero_correction_preserves_rasterizer_view_matrix_convention(self):
        base = self.angle2pose(5.0, np.deg2rad(37.0), [0.13, -0.07])
        corrected = self.corrected(base, self.empty_values())
        stored_view = np.linalg.inv(corrected).T

        np.testing.assert_allclose(corrected, base, atol=1e-12)
        # CUDA's transformPoint4x3 indexes the contiguous transposed matrix
        # as W2C, matching the kernel's column-major indexing convention.
        np.testing.assert_allclose(stored_view.T, np.linalg.inv(base), atol=1e-12)

    def test_cone_rotation_quaternion_round_trip_across_scan_angles(self):
        methods = self.pose_functions
        matrix_to_quaternion = methods["_rotation_matrix_to_quaternion"]
        quaternion_to_matrix = methods["_quaternion_to_rotation_matrix"]
        for angle in np.linspace(0.0, 2.0 * np.pi, 721):
            # The cone path converts the camera's W2C rotation to a quaternion.
            w2c_rotation = self.angle2pose(5.0, angle)[:3, :3].T
            quaternion = matrix_to_quaternion(w2c_rotation)
            reconstructed = quaternion_to_matrix(quaternion)
            np.testing.assert_allclose(reconstructed, w2c_rotation, atol=1e-7)

    def test_centered_per_view_offsets_remain_zero_mean_and_bounded(self):
        center_and_bound = production_center_and_bound()
        maximum = 0.5
        angle_offsets = center_and_bound(np.array([maximum, -maximum, -maximum]), maximum)
        shift_offsets = center_and_bound(
            np.array([[maximum, -maximum], [-maximum, maximum], [-maximum, -maximum]]),
            maximum,
        )

        self.assertAlmostEqual(float(angle_offsets.mean()), 0.0, places=12)
        self.assertLessEqual(float(np.abs(angle_offsets).max()), maximum)
        np.testing.assert_allclose(shift_offsets.mean(axis=0), [0.0, 0.0], atol=1e-12)
        self.assertTrue(np.all(np.abs(shift_offsets) <= maximum))

    def test_saved_pose_has_version_and_complete_geometry_report(self):
        class TensorLike:
            def __init__(self, value):
                self.value = np.asarray(value)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class PoseLike:
            def values(self):
                return {
                    "delta_angle": TensorLike([0.01, -0.01]),
                    "global_shift": TensorLike([0.2, -0.3]),
                    "view_shift": TensorLike([[0.1, 0.2], [-0.1, -0.2]]),
                    "beam_tilt": TensorLike([0.003, -0.004]),
                    "detector_roll": TensorLike([0.005]),
                }

        with tempfile.TemporaryDirectory() as directory:
            scene = SimpleNamespace(
                model_path=directory,
                train_cameras=[
                    SimpleNamespace(image_name="view0"),
                    SimpleNamespace(image_name="view1"),
                ],
            )
            production_save_pose_result()(scene, PoseLike())
            output = Path(directory) / "pose_calibration"
            with np.load(output / "pose_offsets.npz") as saved:
                self.assertEqual(int(saved["pose_format_version"].item()), 2)
                np.testing.assert_allclose(saved["detector_roll"], [0.005])
            with (output / "pose_report.csv").open(newline="", encoding="utf-8") as handle:
                report = list(csv.DictReader(handle))

        self.assertEqual(len(report), 2)
        self.assertEqual(
            set(report[0]),
            {
                "view",
                "image",
                "angle_delta_rad",
                "shift_u_px",
                "shift_v_px",
                "beam_tilt_u_rad",
                "beam_tilt_v_rad",
                "detector_roll_rad",
            },
        )
        self.assertAlmostEqual(float(report[0]["shift_u_px"]), 0.3)


if __name__ == "__main__":
    unittest.main()
