"""FBP backends used exclusively by scripts/apply_pose_fbp.py.

Full mode filters in the physical scan-tangent direction and backprojects with
the complete ray geometry. It is a uniform-quadrature, limited-trajectory
laminographic FBP approximation, not an iterative reconstruction or a promise
of an exact inverse for incomplete data.
"""

import importlib
import inspect
from functools import lru_cache

import numpy as np
from scipy import fft
from scipy.ndimage import map_coordinates

from r2_gaussian.utils.pose_fbp_geometry import build_geometry, project_points, validate_geometry


FILTERS = ("ram_lak", "shepp_logan", "cosine", "hamming", "hann")


def _filter_window(frequency, name):
    # frequency is in cycles per detector sample, so Nyquist is 0.5.
    if name not in FILTERS:
        raise ValueError(f"Unsupported FBP filter: {name}")
    if name == "ram_lak":
        return np.ones_like(frequency)
    if name == "shepp_logan":
        return np.sinc(frequency)
    if name == "cosine":
        return np.cos(np.pi * frequency)
    if name == "hamming":
        return 0.54 + 0.46 * np.cos(2 * np.pi * frequency)
    return (1 + np.cos(2 * np.pi * frequency)) / 2


def _fft_length(count):
    # A fixed padding rule shared by both branches and all installed backends.
    return max(64, 1 << (2 * int(count)).bit_length())


def _ramp_response(length):
    # Exact frequency-domain ramp with zero DC. Truncating a spatial ramp
    # instead leaves a padding-dependent DC term and breaks the 1D/2D limit
    # when detector height exceeds width.
    return 2 * np.fft.rfftfreq(length)


def filter_projection(projection, pixel_size_vu, direction_uv, n_views, filter_name="ram_lak"):
    """Directional frequency ramp with physical uniform normalization.

    The ray sweep derivative is z_axis cross ray. Its U/V components (a,b)
    give a physical multiplier |a*f_u+b*f_v|. Its norm supplies the inclination
    factor. On the standard orbit this reduces to the existing 1D ramp with
    scale pi/(2*N*pixel_u). Every branch uses the same zero-DC definition,
    independently of the installed TIGRE filter implementation.
    The 2D case is zero-padded on both axes to avoid circular wraparound.
    """
    image = np.asarray(projection, dtype=np.float32)
    pitch = np.asarray(pixel_size_vu, dtype=np.float64)
    direction = np.asarray(direction_uv, dtype=np.float64)
    if (image.ndim != 2 or not np.isfinite(image).all() or pitch.shape != (2,)
            or not np.isfinite(pitch).all() or np.any(pitch <= 0)
            or direction.shape != (2,) or not np.isfinite(direction).all() or n_views <= 0):
        raise ValueError("Invalid directional filter input")
    magnitude = float(np.linalg.norm(direction))
    if magnitude < 1e-8:
        raise ValueError("Degenerate scan tangent for FBP")
    unit_u, unit_v = direction / magnitude
    height, width = image.shape
    length_u = _fft_length(width)
    start_u = (length_u - width) // 2
    if abs(unit_v) < 1e-12:
        response = _ramp_response(length_u)
        response *= _filter_window(np.fft.rfftfreq(length_u), filter_name)
        padded = np.zeros((height, length_u), np.float32)
        padded[:, start_u:start_u + width] = image
        result = fft.irfft(fft.rfft(padded, axis=1) * response, n=length_u, axis=1)
        result = result[:, start_u:start_u + width]
        return np.asarray(result * (magnitude * np.pi / (2 * n_views * pitch[1])), np.float32)

    length_v = _fft_length(height)
    start_v = (length_v - height) // 2
    padded = np.zeros((length_v, length_u), np.float32)
    padded[start_v:start_v + height, start_u:start_u + width] = image
    frequency_u = np.fft.rfftfreq(length_u, d=pitch[1])[None, :]
    frequency_v = np.fft.fftfreq(length_v, d=pitch[0])[:, None]
    directional_frequency = np.abs(unit_u * frequency_u + unit_v * frequency_v)
    # Maximum directional frequency over the rectangular 2D FFT domain is
    # |unit_u|/(2*du)+|unit_v|/(2*dv). This choice makes window support vary
    # continuously at either detector axis, without a hard Nyquist jump.
    sample_pitch = 1 / (abs(unit_u) / pitch[1] + abs(unit_v) / pitch[0])
    normalized = directional_frequency * sample_pitch
    normalized = np.minimum(normalized, 0.5)
    response = 2 * normalized
    response *= _filter_window(normalized, filter_name)
    result = fft.irfft2(fft.rfft2(padded) * response, s=padded.shape)
    result = result[start_v:start_v + height, start_u:start_u + width]
    return np.asarray(result * (magnitude * np.pi / (2 * n_views * sample_pitch)), np.float32)


def filter_projections(projections, geometry, filter_name="ram_lak"):
    validate_geometry(geometry)
    projections = np.asarray(projections, dtype=np.float32)
    expected = (len(geometry["scan_angles"]), *geometry["nDetector"])
    if projections.shape != expected:
        raise ValueError(f"Projection stack must have shape {expected}")
    filtered = np.empty_like(projections)
    for index, projection in enumerate(projections):
        filtered[index] = filter_projection(projection, geometry["pixel_size_vu"],
                                             geometry["filter_direction_uv"][index],
                                             len(projections), filter_name)
    return filtered


def voxel_points(scanner, start, stop):
    shape = tuple(scanner["nVoxel"])
    indices = np.column_stack(np.unravel_index(np.arange(start, stop), shape))
    return (np.asarray(scanner["offOrigin"])
            + ((indices + 0.5) / np.asarray(shape) - 0.5) * np.asarray(scanner["sVoxel"]))


def backproject_cpu(filtered, geometry, scanner, chunk_voxels=262144):
    """Small-volume CPU reference; never selected as an automatic fallback."""
    if chunk_voxels < 1:
        raise ValueError("chunk_voxels must be positive")
    volume = np.zeros(tuple(scanner["nVoxel"]), dtype=np.float32)
    flat = volume.ravel()
    for start in range(0, flat.size, chunk_voxels):
        stop = min(start + chunk_voxels, flat.size)
        points = voxel_points(scanner, start, stop)
        accumulated = np.zeros(stop - start, dtype=np.float64)
        for index, projection in enumerate(filtered):
            uv = project_points(points, geometry, index)
            # CUDA border textures interpolate partially supported edge pixels.
            accumulated += map_coordinates(projection, [uv[:, 1], uv[:, 0]],
                                           order=1, mode="grid-constant", cval=0, prefilter=False)
        flat[start:stop] = accumulated
    return volume


def require_backend(backend):
    if backend == "cpu":
        return
    if backend != "tigre":
        raise ValueError(f"Unsupported backend: {backend}")
    try:
        importlib.import_module("tigre.algorithms")
        importlib.import_module("r2_gaussian.utils.ct_utils")
    except ImportError as error:
        raise RuntimeError(
            "The TIGRE backend requires the project's PyTorch/TIGRE environment. "
            "Use --export-only to export inputs, or explicitly use --backend cpu for a small-volume reference."
        ) from error


def _geometry_for_tigre(geometry, scanner, layout):
    from r2_gaussian.utils.ct_utils import get_geometry_tigre

    geo = get_geometry_tigre(scanner)
    # Python package sources need not match the installed compiled extension.
    # Only the offline adapter changes encoding; world XYZ and physical U/V
    # vectors in the saved geometry remain identical for both ABI layouts.
    origin = np.asarray(scanner["offOrigin"], dtype=np.float32)
    detector_offsets = np.asarray(geometry["tigre_offset_uv"], dtype=np.float32)
    geo.offOrigin = origin if layout["off_origin_order"] == "xyz" else origin[::-1].copy()
    from r2_gaussian.utils.geometry_utils import reorder_tigre_detector_offset
    geo.offDetector = reorder_tigre_detector_offset(
        detector_offsets, layout["off_detector_order"]
    )
    count = len(geometry["scan_angles"])
    geo.rotDetector = np.zeros((count, 3), dtype=np.float32)
    geo.COR = np.zeros(count, dtype=np.float32)
    return geo


def _tigre_backproject(filtered, geometry, scanner, layout=None):
    from tigre.utilities.Atb import Atb

    standard_orbit = np.all(geometry["beam_tilt"] == 0)
    if layout is None:
        probe = probe_tigre_standard_geometry if standard_orbit else probe_tigre_full_geometry
        layout = probe()["layout"]
    geo = _geometry_for_tigre(geometry, scanner, layout)
    angle_values = geometry["scan_angles"] if standard_orbit else geometry["tigre_angles"]
    angles = np.ascontiguousarray(angle_values, dtype=np.float32)
    # Angle-dependent normalization is already included in our filter. Atb is
    # an unweighted sum here, so do NOT apply TIGRE FBP's DSO/DSD again.
    result = Atb(np.ascontiguousarray(filtered[:, ::-1, :]), geo, angles)
    return np.asarray(result.transpose(2, 1, 0), dtype=np.float32)


@lru_cache(maxsize=1)
def probe_tigre_full_geometry():
    return _probe_tigre_geometry(full_pose=True)


@lru_cache(maxsize=1)
def probe_tigre_standard_geometry():
    """Check ordinary 1D-angle FBP without requiring arbitrary 3D rotations."""
    return _probe_tigre_geometry(full_pose=False)


def _probe_tigre_geometry(full_pose):
    """Identify offset encoding using actual coordinate-valued backprojections.

    Known TIGRE builds differ in Python/CUDA XYZ versus ZYX origin encoding
    and UV versus VU detector-offset encoding. Test all four combinations on
    several non-cubic, offset, tilted acquisitions. U and V are checked
    separately in PIXELS, so cancellation in a combined ramp cannot hide a
    geometry error. The fixed tolerance includes texture interpolation's
    1/256 fractional-coordinate quantization, not an arbitrary intensity error.
    """
    scanner = dict(mode="parallel", DSO=5.0, DSD=10.0, nVoxel=[3, 4, 5],
                   sVoxel=[0.7, 0.8, 0.9], offOrigin=[0.01, -0.02, 0.03],
                   nDetector=[32, 40], sDetector=[3.2, 5.0], offDetector=[0.02, -0.03],
                   accuracy=0.5, filter=None)
    vv, uu = np.indices(scanner["nDetector"])
    ramps = [uu.astype(np.float32)[None], vv.astype(np.float32)[None]]
    cases = []
    settings = ((0.17, [0, 0, 0], [0.02, -0.03], [0.12, -0.09]),
                (0.67, [0.01, -0.02, 0.03], [0.02, -0.03], [0.12, -0.09]),
                (1.31, [-0.04, 0.01, 0.02], [-0.05, 0.04], [-0.07, 0.04]))
    for angle, origin, detector_offset, tilt in settings:
        cfg = {**scanner, "offOrigin": origin, "offDetector": detector_offset}
        geometry = build_geometry(cfg, [angle], tilt if full_pose else [0, 0])
        points = voxel_points(cfg, 0, int(np.prod(cfg["nVoxel"])))
        uv = project_points(points, geometry, 0)
        # All reference samples must have full support: this is a coordinate
        # probe, not a test of CUDA's partially supported border textures.
        if np.any(uv < 1) or np.any(uv > np.asarray(cfg["nDetector"])[::-1] - 2):
            raise RuntimeError("Internal TIGRE probe coordinates leave the detector interior")
        cases.append((cfg, geometry, uv))

    tolerance_px, gain_tolerance = 0.005, 1e-5
    candidates = []
    for origin_order in ("zyx", "xyz"):
        for detector_order in ("uv", "vu"):
            layout = {"off_origin_order": origin_order, "off_detector_order": detector_order}
            case_errors, gain_error = [], 0.0
            for cfg, geometry, uv in cases:
                errors = []
                for axis, ramp in enumerate(ramps):
                    actual = _tigre_backproject(ramp, geometry, cfg, layout=layout)
                    if actual.shape != tuple(cfg["nVoxel"]) or not np.isfinite(actual).all():
                        raise RuntimeError("Installed TIGRE returned invalid geometry-probe data; no partial fallback")
                    errors.append(float(np.max(np.abs(actual.ravel() - uv[:, axis]))))
                constant = _tigre_backproject(np.ones_like(ramps[0]), geometry, cfg, layout=layout)
                if not np.isfinite(constant).all():
                    raise RuntimeError("Installed TIGRE returned non-finite probe normalization")
                gain_error = max(gain_error, float(np.max(np.abs(constant - 1))))
                case_errors.append({"u_px": errors[0], "v_px": errors[1]})
            candidates.append({"layout": layout, "case_errors_px": case_errors,
                               "max_coordinate_error_px": max(max(row.values()) for row in case_errors),
                               "constant_gain_error": gain_error})
    accepted = [row for row in candidates if row["max_coordinate_error_px"] <= tolerance_px
                and row["constant_gain_error"] <= gain_tolerance]
    if len(accepted) != 1:
        diagnostics = "; ".join(
            f"origin={row['layout']['off_origin_order']},detector={row['layout']['off_detector_order']}: "
            f"{row['max_coordinate_error_px']:.6g}px,gain={row['constant_gain_error']:.6g}"
            for row in candidates)
        kind = "full-pose" if full_pose else "standard"
        raise RuntimeError(f"Installed TIGRE failed {kind} geometry probe "
                           f"({len(accepted)} valid layouts; tolerance {tolerance_px}px; {diagnostics}); no partial fallback")
    best = accepted[0]
    return {"status": "passed", "geometry_kind": "full_pose" if full_pose else "standard",
            **best, "tolerance_px": tolerance_px,
            "gain_tolerance": gain_tolerance, "candidate_scores": candidates}


def reconstruct_standard(projections, angles, scanner, filter_name="ram_lak", backend="tigre"):
    require_backend(backend)
    geometry = build_geometry(scanner, angles)
    filtered = filter_projections(projections, geometry, filter_name)
    if backend == "cpu":
        return backproject_cpu(filtered, geometry, scanner)
    # Use the SAME physical filter as full mode. Installed algs.fbp versions
    # can ignore filter=, normalize by detector V pitch while filtering U, or
    # choose different padding. Keep that variability out of this offline tool.
    return _tigre_backproject(filtered, geometry, scanner)


def reconstruct_full(projections, geometry, scanner, filter_name="ram_lak", backend="tigre"):
    validate_geometry(geometry)
    require_backend(backend)
    # The zero-tilt limit uses the exact standard implementation, not a second
    # interpolation/filter implementation with a slightly different baseline.
    if np.all(geometry["beam_tilt"] == 0):
        return reconstruct_standard(projections, geometry["scan_angles"], scanner, filter_name, backend)
    filtered = filter_projections(projections, geometry, filter_name)
    if backend == "cpu":
        return backproject_cpu(filtered, geometry, scanner)
    return _tigre_backproject(filtered, geometry, scanner)


def software_versions():
    from importlib.metadata import PackageNotFoundError, version
    import platform

    result = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "matplotlib", "torch", "pytigre", "tigre"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def backend_provenance(backend):
    """Record the installed Python wrapper identity when distributions lack metadata."""
    if backend == "cpu":
        return {"backend": "cpu", "reference_only": True}
    from r2_gaussian.utils.artifact_utils import file_identity
    from tigre.utilities.Atb import Atb

    result = {"backend": "tigre", "filter_backend": "offline_physical_frequency_ramp"}
    for name, function in (("Atb", Atb),):
        path = inspect.getsourcefile(function)
        result[name] = file_identity(path) if path else {"source": "unavailable"}
    return result
