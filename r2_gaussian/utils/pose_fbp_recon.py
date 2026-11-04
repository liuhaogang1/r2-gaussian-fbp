"""FBP backends used exclusively by scripts/apply_pose_fbp.py.

Full mode filters in the physical scan-tangent direction and backprojects with
the complete ray geometry. It is a uniform-quadrature, limited-trajectory
laminographic FBP approximation, not an iterative reconstruction or a promise
of an exact inverse for incomplete data.
"""

import importlib
import inspect

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
    # Match TIGRE's strict next-power-of-two padding for the zero-tilt limit.
    return max(64, 1 << (2 * int(count)).bit_length())


def _ramp_response(length):
    positions = np.arange(-length // 2, length // 2, dtype=np.float64)
    kernel = np.zeros(length, dtype=np.float64)
    kernel[length // 2] = 0.25
    odd = (positions.astype(np.int64) % 2) != 0
    kernel[odd] = -1 / (np.pi * positions[odd]) ** 2
    return 2 * np.abs(np.fft.rfft(kernel))


def filter_projection(projection, pixel_size_vu, direction_uv, n_views, filter_name="ram_lak"):
    """Directional discrete ramp with TIGRE-compatible uniform normalization.

    The ray sweep derivative is z_axis cross ray. Its U/V components (a,b)
    give a physical multiplier |a*f_u+b*f_v|. Its norm supplies the inclination
    factor. On the standard orbit this reduces to the existing 1D ramp with
    scale pi/(2*N*pixel_u), including the finite discrete ramp's DC response.
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
    sample_pitch = max(abs(unit_u) * pitch[1], abs(unit_v) * pitch[0])
    normalized = directional_frequency * sample_pitch
    table_length = max(length_u, length_v)
    response = np.interp(normalized, np.fft.rfftfreq(table_length), _ramp_response(table_length))
    response *= _filter_window(normalized, filter_name)
    response[normalized > 0.5] = 0
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


def _tigre_backproject(filtered, geometry, scanner):
    from tigre.utilities.Atb import Atb
    from r2_gaussian.utils.ct_utils import get_geometry_tigre

    geo = get_geometry_tigre(scanner)
    geo.offDetector = np.asarray(geometry["tigre_offset_uv"], dtype=np.float32)
    geo.rotDetector = np.zeros((len(filtered), 3), dtype=np.float32)
    geo.COR = np.zeros(len(filtered), dtype=np.float32)
    angles = np.ascontiguousarray(geometry["tigre_angles"], dtype=np.float32)
    # Angle-dependent normalization is already included in our filter. Atb is
    # an unweighted sum here, so do NOT apply TIGRE FBP's DSO/DSD again.
    result = Atb(np.ascontiguousarray(filtered[:, ::-1, :]), geo, angles)
    return np.asarray(result.transpose(2, 1, 0), dtype=np.float32)


def probe_tigre_full_geometry():
    """Check the INSTALLED CUDA kernel, not just acceptance of (N,3) angles."""
    scanner = dict(mode="parallel", DSO=5.0, DSD=10.0, nVoxel=[3, 4, 5],
                   sVoxel=[0.7, 0.8, 0.9], offOrigin=[0.01, -0.02, 0.03],
                   nDetector=[32, 40], sDetector=[3.2, 4.0], offDetector=[0.02, -0.03],
                   accuracy=0.5, filter=None)
    geometry = build_geometry(scanner, [0.67], [0.12, -0.09])
    vv, uu = np.indices(scanner["nDetector"])
    ramp = (3 + 0.2 * uu + 0.3 * vv).astype(np.float32)[None]
    points = voxel_points(scanner, 0, int(np.prod(scanner["nVoxel"])))
    uv = project_points(points, geometry, 0)
    expected = (3 + 0.2 * uv[:, 0] + 0.3 * uv[:, 1]).reshape(scanner["nVoxel"])
    actual = _tigre_backproject(ramp, geometry, scanner)
    error = float(np.max(np.abs(actual - expected)))
    # Hardware bilinear texture fractions are quantized; this is far smaller
    # than the discrepancy from ignoring tilt, an offset, or flipping twice.
    if not np.isfinite(actual).all() or error > 0.02:
        raise RuntimeError(f"Installed TIGRE failed full-pose geometry probe (max error {error:.6g}); no partial fallback")
    return {"status": "passed", "max_ramp_error": error, "tolerance": 0.02}


def reconstruct_standard(projections, angles, scanner, filter_name="ram_lak", backend="tigre"):
    if backend == "cpu":
        geometry = build_geometry(scanner, angles)
        return backproject_cpu(filter_projections(projections, geometry, filter_name), geometry, scanner)
    require_backend(backend)
    import tigre.algorithms as algorithms
    from r2_gaussian.utils.ct_utils import get_geometry_tigre

    # Preserve the main path's V reversal and XYZ result convention. Pass the
    # filter explicitly: upstream FBP can overwrite geo.filter with its default.
    result = algorithms.fbp(np.ascontiguousarray(projections[:, ::-1, :], dtype=np.float32),
                            get_geometry_tigre(scanner), np.asarray(angles, dtype=np.float32),
                            filter=filter_name)
    return np.asarray(result.transpose(2, 1, 0), dtype=np.float32)


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
    import tigre.algorithms as algorithms

    result = {"backend": "tigre"}
    for name, function in (("fbp", algorithms.fbp), ("Atb", Atb)):
        path = inspect.getsourcefile(function)
        result[name] = file_identity(path) if path else {"source": "unavailable"}
    return result
