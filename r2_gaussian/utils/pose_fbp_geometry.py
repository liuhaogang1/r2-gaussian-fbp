"""Physical parallel-beam geometry for the optional offline pose/FBP tool.

Images have [V,U] axes, shifts have [U,V] axes, and world/volume axes are XYZ.
The exported geometry describes images AFTER learned shift and roll removal.
No training modules, torch, or TIGRE are imported here.
"""

import numpy as np
from scipy.spatial.transform import Rotation


GEOMETRY_VERSION = 1


def rotation(axis, angle):
    vector = np.zeros(3, dtype=np.float64)
    vector[axis] = angle
    return Rotation.from_rotvec(vector).as_matrix()


def camera_pose(scanner, angle, beam_tilt=(0.0, 0.0)):
    """Match angle2pose and Camera._correct_parallel_c2w in physical units.

    Scan angle rotates the complete pose, including its offset center. Local
    beam rotations then change its axes while keeping that center fixed.
    Detector roll is absent because it has already been removed from pixels.
    """
    base = rotation(2, angle) @ rotation(2, np.pi / 2) @ rotation(0, -np.pi / 2)
    offset = np.asarray(scanner.get("offDetector", [0, 0]), dtype=np.float64)
    center = float(scanner["DSO"]) * np.array([np.cos(angle), np.sin(angle), 0.0])
    center += base[:, :2] @ offset
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = base @ rotation(0, beam_tilt[0]) @ rotation(1, beam_tilt[1])
    result[:3, 3] = center
    return result


def tigre_rotation_matrices(angles):
    """World-to-TIGRE frames used by CUDA eulerZYZT, with negated inputs.

    CUDA evaluates Rz(-psi) Ry(-theta) Rz(-alpha), in that order.
    Keep this explicit: SciPy's ZYZ component order is different.
    """
    return np.asarray([
        rotation(2, -psi) @ rotation(1, -theta) @ rotation(2, -alpha)
        for alpha, theta, psi in np.asarray(angles, dtype=np.float64)
    ])


def tigre_angles_from_frames(frames):
    """Decompose world-to-TIGRE frames without discarding near-singular tilt.

    For M = Rz(first) Ry(middle) Rz(last), atan2(hypot(M02,M12),M22)
    retains small middle angles even when M22 has rounded to exactly one.
    SciPy's as_euler uses a finite gimbal-lock threshold and may collapse
    rotations below it, which is incompatible with our strict roundtrip check.

    The first/last angles are not individually stable near middle=0 or pi.
    Recover their well-conditioned SUM near zero and DIFFERENCE near pi
    from the upper-left block, rather than independently dividing tiny entries.
    This also handles the exact singularities; no nonzero tilt is zeroed.
    """
    matrices = np.asarray(frames, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError("TIGRE camera frames must be finite (N,3,3) matrices")
    middle = np.arctan2(np.hypot(matrices[:, 0, 2], matrices[:, 1, 2]), matrices[:, 2, 2])
    first = np.arctan2(matrices[:, 1, 2], matrices[:, 0, 2])
    summed = np.arctan2(matrices[:, 1, 0] - matrices[:, 0, 1],
                        matrices[:, 0, 0] + matrices[:, 1, 1])
    difference = np.arctan2(-matrices[:, 1, 0] - matrices[:, 0, 1],
                            matrices[:, 1, 1] - matrices[:, 0, 0])
    last = np.where(matrices[:, 2, 2] >= 0, summed - first, first - difference)
    # CUDA evaluates Rz(-psi) Ry(-theta) Rz(-alpha).
    return -np.column_stack((last, middle, first))


def build_geometry(scanner, angles, beam_tilt=(0.0, 0.0)):
    """Return serializable ray geometry and its verified TIGRE adaptation."""
    angles = np.asarray(angles, dtype=np.float64)
    tilt = np.asarray(beam_tilt, dtype=np.float64)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError("Scan angles must be a nonempty finite vector")
    if tilt.shape != (2,) or not np.isfinite(tilt).all():
        raise ValueError("beam_tilt must contain two finite radians")
    poses = np.stack([camera_pose(scanner, angle, tilt) for angle in angles])
    u, v, ray = (poses[:, :3, axis] for axis in range(3))
    centers = poses[:, :3, 3]
    # After the existing saved-row reversal, TIGRE detector V points along -v.
    frames = np.stack((-ray, u, -v), axis=1)
    tigre_angles = tigre_angles_from_frames(frames)
    if np.all(tilt == 0):
        # Use exactly the same representation as the ordinary scalar-angle path.
        tigre_angles = np.column_stack((angles, np.zeros((len(angles), 2))))
    if not np.allclose(tigre_rotation_matrices(tigre_angles), frames, atol=1e-10, rtol=0):
        errors = np.max(np.abs(tigre_rotation_matrices(tigre_angles) - frames), axis=(1, 2))
        index = int(np.argmax(errors))
        raise ValueError("Cannot represent the camera frame in TIGRE ZYZ coordinates "
                         f"(max matrix error {errors[index]:.6g}, view {index}, beam_tilt={tilt.tolist()})")
    # Tilt rotates about the original camera center. Project its FULL position,
    # including DSO and static offDetector, into the new detector basis.
    offsets = np.column_stack((np.sum(centers * u, axis=1), -np.sum(centers * v, axis=1)))
    # d(ray)/d(scan angle) = world_Z cross ray. Its detector components define
    # the directional ramp and its inclination factor (laminographic FBP).
    tangent = np.cross(np.array([0.0, 0.0, 1.0]), ray)
    direction = np.column_stack((np.sum(tangent * u, axis=1), np.sum(tangent * v, axis=1)))
    if np.any(np.linalg.norm(direction, axis=1) < 1e-8):
        raise ValueError("Ray directions parallel to the rotation axis cannot use this FBP")
    geometry = {
        "geometry_version": np.asarray(GEOMETRY_VERSION, dtype=np.int32),
        "camera_c2w": poses,
        "ray_direction": ray,
        "u_axis": u,
        "v_axis": v,
        "detector_center": centers,
        "nDetector": np.asarray(scanner["nDetector"], dtype=np.int64),
        "pixel_size_vu": np.asarray(scanner["sDetector"]) / np.asarray(scanner["nDetector"]),
        "scan_angles": angles,
        "beam_tilt": tilt,
        "tigre_angles": tigre_angles,
        "tigre_offset_uv": offsets,
        "filter_direction_uv": direction,
    }
    validate_geometry(geometry)
    return geometry


def validate_geometry(geometry):
    """Reject corrupt or inconsistent saved geometry before reconstruction."""
    if np.asarray(geometry["geometry_version"]).shape != () or int(geometry["geometry_version"]) != GEOMETRY_VERSION:
        raise ValueError("Unsupported full-pose geometry version")
    angles = np.asarray(geometry["scan_angles"])
    if angles.ndim != 1 or not len(angles):
        raise ValueError("Invalid geometry angle vector")
    n = len(angles)
    shapes = {"camera_c2w": (n, 4, 4), "ray_direction": (n, 3), "u_axis": (n, 3),
              "v_axis": (n, 3), "detector_center": (n, 3), "nDetector": (2,),
              "pixel_size_vu": (2,), "beam_tilt": (2,), "tigre_angles": (n, 3),
              "tigre_offset_uv": (n, 2), "filter_direction_uv": (n, 2), "scan_angles": (n,)}
    for name, shape in shapes.items():
        value = np.asarray(geometry[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid saved geometry array: {name}")
    counts = geometry["nDetector"]
    if np.any(counts <= 0) or np.any(counts != np.floor(counts)) or np.any(geometry["pixel_size_vu"] <= 0):
        raise ValueError("Invalid saved detector dimensions or pitch")
    poses = geometry["camera_c2w"]
    rotations = poses[:, :3, :3]
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-9, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-9, rtol=0)
            or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-12, rtol=0)):
        raise ValueError("Saved camera frames are not rigid transforms")
    u, v, ray = (geometry[key] for key in ("u_axis", "v_axis", "ray_direction"))
    if (not np.allclose(np.stack((u, v, ray), axis=2), rotations, atol=1e-10, rtol=0)
            or not np.allclose(poses[:, :3, 3], geometry["detector_center"], atol=1e-10, rtol=0)):
        raise ValueError("Saved ray vectors disagree with camera transforms")
    frames = np.stack((-ray, u, -v), axis=1)
    offsets = np.column_stack((np.sum(u * geometry["detector_center"], axis=1),
                               -np.sum(v * geometry["detector_center"], axis=1)))
    tangent = np.cross([0, 0, 1], ray)
    directions = np.column_stack((np.sum(tangent * u, axis=1), np.sum(tangent * v, axis=1)))
    if (not np.allclose(tigre_rotation_matrices(geometry["tigre_angles"]), frames, atol=1e-9, rtol=0)
            or not np.allclose(geometry["tigre_offset_uv"], offsets, atol=1e-9, rtol=0)
            or not np.allclose(geometry["filter_direction_uv"], directions, atol=1e-9, rtol=0)
            or np.any(np.linalg.norm(directions, axis=1) < 1e-8)):
        raise ValueError("Saved TIGRE adaptation or filter direction is inconsistent")


def load_geometry(path):
    with np.load(path, allow_pickle=False) as data:
        geometry = {name: data[name] for name in data.files}
    validate_geometry(geometry)
    return geometry


def project_points(points, geometry, index):
    """Project physical XYZ points into saved-image [U,V] pixel coordinates."""
    local = np.asarray(points, dtype=np.float64) - geometry["detector_center"][index]
    uv = np.column_stack((local @ geometry["u_axis"][index], local @ geometry["v_axis"][index]))
    return uv / geometry["pixel_size_vu"][::-1] + (geometry["nDetector"][::-1] - 1) / 2
