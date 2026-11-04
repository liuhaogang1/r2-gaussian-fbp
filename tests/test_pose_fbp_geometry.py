"""Production-camera equivalence and independent analytic phantom checks."""

import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from test_parallel_pose_geometry import production_angle2pose, production_pose_functions
from r2_gaussian.utils.pose_fbp import correction_affine
from r2_gaussian.utils.pose_fbp_geometry import (build_geometry, project_points,
                                                tigre_angles_from_frames, tigre_rotation_matrices,
                                                validate_geometry)
from r2_gaussian.utils.pose_fbp_recon import (filter_projection, probe_tigre_full_geometry,
                                             reconstruct_full, reconstruct_standard, voxel_points)


def scanner():
    return dict(mode="parallel", DSO=3.0, DSD=6.0, nDetector=[48, 64],
                sDetector=[2.16, 2.56], nVoxel=[31, 29, 27], sVoxel=[1.8, 1.7, 1.6],
                offOrigin=[0.04, -0.03, 0.02], offDetector=[0.07, -0.04], accuracy=0.5, filter=None)


def production_camera(cfg, angle, delta=0.0, tilt=(0, 0), roll=0.0):
    methods = production_pose_functions()
    base = production_angle2pose()(cfg["DSO"], angle, cfg["offDetector"])
    camera = SimpleNamespace(_base_world_view_transform=np.linalg.inv(base).T, pose_index=0,
                             _rot_x=methods["_rot_x"], _rot_y=methods["_rot_y"], _rot_z=methods["_rot_z"])
    values = dict(delta_angle=np.array([delta]), beam_tilt=np.asarray(tilt), detector_roll=np.array([roll]))
    return methods["_correct_parallel_c2w"](camera, values, "cpu")


def analytic_data(cfg, angles, tilt):
    """Line integrals through 3D Gaussians using the PRODUCTION camera frame.

    This generator uses neither the new geometry conversion nor an image warp;
    it integrates a known physical density along the actual acquisition rays.
    """
    centers = np.array([[0.23, 0.16, -0.19], [-0.27, -0.22, 0.24]])
    sigma, amplitudes = 0.105, np.array([1.0, 0.75])
    vv, uu = np.indices(cfg["nDetector"])
    pitch = np.asarray(cfg["sDetector"]) / cfg["nDetector"]
    centered_u = (uu - (uu.shape[1] - 1) / 2) * pitch[1]
    centered_v = (vv - (vv.shape[0] - 1) / 2) * pitch[0]
    projections = []
    for angle in angles:
        camera = production_camera(cfg, angle, tilt=tilt)
        detector = (camera[:3, 3] + centered_u[..., None] * camera[:3, 0]
                    + centered_v[..., None] * camera[:3, 1])
        image = np.zeros(vv.shape, np.float64)
        for center, amplitude in zip(centers, amplitudes):
            difference = detector - center
            perpendicular = difference - np.sum(difference * camera[:3, 2], axis=-1)[..., None] * camera[:3, 2]
            image += amplitude * np.sqrt(2 * np.pi) * sigma * np.exp(-np.sum(perpendicular ** 2, axis=-1) / (2 * sigma ** 2))
        projections.append(image)
    points = voxel_points(cfg, 0, int(np.prod(cfg["nVoxel"])))
    truth = sum(amplitude * np.exp(-np.sum((points - center) ** 2, axis=1) / (2 * sigma ** 2))
                for center, amplitude in zip(centers, amplitudes)).reshape(cfg["nVoxel"])
    return np.asarray(projections, np.float32), truth


def has_tigre():
    try:
        if not all(importlib.util.find_spec(name) is not None for name in ("torch", "tigre")):
            return False
        import torch
        return torch.cuda.is_available()
    except (ValueError, ImportError):
        return False


class FullGeometryTests(unittest.TestCase):
    def test_sub_microradian_tilts_preserve_camera_frames_across_scan_angles(self):
        cfg = scanner()
        angles = np.linspace(-np.pi, 3 * np.pi, 37)
        for scale in (1e-12, 1e-10, 1e-9, 1e-8, 3e-8, 7e-8, 1e-7, 3e-7, 1e-6):
            for components in ((1, -0.7), (-1, 0.7), (0, 1), (1, 0)):
                tilt = scale * np.array(components)
                with self.subTest(tilt=tilt):
                    geometry = build_geometry(cfg, angles, tilt)
                    reconstructed = tigre_rotation_matrices(geometry["tigre_angles"])
                    expected = np.stack((-geometry["ray_direction"], geometry["u_axis"],
                                         -geometry["v_axis"]), axis=1)
                    np.testing.assert_allclose(reconstructed, expected, atol=3e-15, rtol=0)
                    np.testing.assert_array_equal(geometry["beam_tilt"], tilt)
                    if components[0] != 0:
                        # Keeping the NPZ value alone is insufficient: the
                        # actual backprojector angle must retain the tilt too.
                        self.assertGreater(float(np.min(np.abs(geometry["tigre_angles"][:, 1]))), scale * 0.9)

    def test_zyz_matrix_conversion_handles_both_singularities_and_general_frames(self):
        triples = [[first, middle, last]
                   for middle in (0, 1e-12, 1e-8, np.pi / 2, np.pi - 1e-8, np.pi - 1e-12, np.pi)
                   for first, last in ((0, 0), (0.7, -1.1), (-np.pi, np.pi), (2.7, 1.9))]
        singular_frames = Rotation.from_euler("ZYZ", triples).as_matrix()
        general_frames = Rotation.from_rotvec(np.random.default_rng(27).normal(size=(200, 3))).as_matrix()
        frames = np.concatenate((singular_frames, general_frames))
        np.testing.assert_allclose(tigre_rotation_matrices(tigre_angles_from_frames(frames)),
                                   frames, atol=3e-15, rtol=0)

    def test_saved_geometry_matches_production_including_nonzero_tilt_offset_and_roll(self):
        cfg = scanner()
        rng = np.random.default_rng(17)
        points = rng.uniform(-0.6, 0.6, (100, 3))
        pitch_uv = (np.asarray(cfg["sDetector"]) / cfg["nDetector"])[::-1]
        center_uv = (np.asarray(cfg["nDetector"])[::-1] - 1) / 2
        shift, roll, delta, tilt = np.array([1.7, -0.8]), 0.008, 0.007, [0.004, -0.006]
        for angle in (0, 0.7, 2.1):
            geometry = build_geometry(cfg, [angle + delta], tilt)
            actual_pose = production_camera(cfg, angle, delta, tilt, roll)
            world = np.column_stack((points, np.ones(len(points))))
            actual_raw = (world @ np.linalg.inv(actual_pose).T)[:, :2] / pitch_uv + center_uv + shift
            corrected_uv = project_points(points, geometry, 0)
            matrix_vu, offset_vu = correction_affine(cfg["nDetector"], pitch_uv[::-1], shift, roll)
            sampled_raw = (corrected_uv[:, ::-1] @ matrix_vu.T + offset_vu)[:, ::-1]
            np.testing.assert_allclose(sampled_raw, actual_raw, atol=2e-11, rtol=0)

    def test_tigre_kernel_equations_project_the_same_points_as_production(self):
        cfg = scanner()
        geometry = build_geometry(cfg, [0.1, 0.9, 2.4], [0.08, -0.05])
        points = np.array([[0.2, -0.4, 0.1], [-0.3, 0.1, -0.2], [0, 0, 0]])
        for index in range(3):
            # Independent expansion of CUDA eulerZYZT (negative input angles).
            alpha, theta, psi = -geometry["tigre_angles"][index]
            ca, sa, ct, st, cp, sp = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
            y = ((sp * ct * ca + cp * sa) * points[:, 0]
                 + (-sp * ct * sa + cp * ca) * points[:, 1] + sp * st * points[:, 2])
            z = -st * ca * points[:, 0] + st * sa * points[:, 1] + ct * points[:, 2]
            uv_tigre = np.column_stack((y, z)) - geometry["tigre_offset_uv"][index]
            uv_tigre = uv_tigre / geometry["pixel_size_vu"][::-1] + (geometry["nDetector"][::-1] - 1) / 2
            uv_tigre[:, 1] = cfg["nDetector"][0] - 1 - uv_tigre[:, 1]
            np.testing.assert_allclose(uv_tigre, project_points(points, geometry, index), atol=1e-10, rtol=0)

    def test_zero_tilt_reduces_to_original_offsets_and_scan_angles(self):
        cfg, angles = scanner(), np.array([0, 0.7, 2.1])
        geometry = build_geometry(cfg, angles)
        np.testing.assert_array_equal(geometry["tigre_angles"][:, 0], angles)
        np.testing.assert_allclose(geometry["tigre_angles"][:, 1:], 0, atol=0)
        np.testing.assert_allclose(geometry["tigre_offset_uv"], np.tile([0.07, 0.04], (3, 1)), atol=1e-13)
        np.testing.assert_allclose(geometry["filter_direction_uv"], np.tile([-1, 0], (3, 1)), atol=1e-13)

    def test_tilt_offsets_include_camera_center_not_just_static_detector_offset(self):
        cfg = scanner()
        a = build_geometry(cfg, [0.4], [0.04, 0.03])
        b = build_geometry({**cfg, "DSO": cfg["DSO"] * 2}, [0.4], [0.04, 0.03])
        self.assertGreater(float(np.linalg.norm(a["tigre_offset_uv"] - b["tigre_offset_uv"])), 0.1)

    def test_corrupt_saved_geometry_is_rejected(self):
        for name in ("u_axis", "tigre_offset_uv", "filter_direction_uv", "camera_c2w"):
            with self.subTest(name=name):
                geometry = build_geometry(scanner(), [0, 0.4], [0.01, 0.02])
                geometry = {key: value.copy() for key, value in geometry.items()}
                geometry[name].flat[0] += 0.1
                with self.assertRaises(ValueError):
                    validate_geometry(geometry)

    def test_directional_filter_rotates_with_detector_and_respects_physical_pitch(self):
        image = np.zeros((25, 25), np.float32)
        image[12, 12] = 1
        horizontal = filter_projection(image, [0.1, 0.1], [1, 0], 30)
        vertical = filter_projection(image, [0.1, 0.1], [0, 1], 30)
        np.testing.assert_allclose(horizontal.T, vertical, atol=1e-8)
        half_sweep = filter_projection(image, [0.1, 0.1], [0.5, 0], 30)
        np.testing.assert_allclose(half_sweep, horizontal / 2, atol=1e-8)
        wider_pixels = filter_projection(image, [0.1, 0.2], [1, 0], 30)
        np.testing.assert_allclose(wider_pixels, horizontal / 2, atol=1e-8)

    def test_full_fbp_improves_independent_tilted_phantom(self):
        cfg = scanner()
        angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        tilt = [0.08, -0.04]
        projections, truth = analytic_data(cfg, angles, tilt)
        geometry = build_geometry(cfg, angles, tilt)
        full = reconstruct_full(projections, geometry, cfg, backend="cpu")
        partial = reconstruct_standard(projections, angles, cfg, backend="cpu")
        full_error = float(np.mean((full - truth) ** 2))
        partial_error = float(np.mean((partial - truth) ** 2))
        self.assertLess(full_error, partial_error * 0.5)
        self.assertAlmostEqual(float(full.max()), float(truth.max()), delta=0.13)
        points = voxel_points(cfg, 0, truth.size).reshape(*truth.shape, 3)
        np.testing.assert_allclose(points[np.unravel_index(full.argmax(), full.shape)],
                                   [0.23, 0.16, -0.19], atol=0.06)

    @unittest.skipUnless(has_tigre(), "Requires installed PyTorch/TIGRE and CUDA")
    def test_installed_tigre_full_geometry_and_cpu_reference_agree(self):
        probe_tigre_full_geometry()
        cfg = scanner()
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        geometry = build_geometry(cfg, angles, [0.06, -0.03])
        projections, _ = analytic_data(cfg, angles, [0.06, -0.03])
        gpu = reconstruct_full(projections, geometry, cfg)
        cpu = reconstruct_full(projections, geometry, cfg, backend="cpu")
        relative = np.linalg.norm(gpu - cpu) / np.linalg.norm(cpu)
        self.assertLess(relative, 0.02)
        zero_geometry = build_geometry(cfg, angles)
        np.testing.assert_array_equal(reconstruct_full(projections, zero_geometry, cfg),
                                      reconstruct_standard(projections, angles, cfg))


if __name__ == "__main__":
    unittest.main()
