"""Input binding and single-resampling exports for optional offline pose FBP."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

from r2_gaussian.utils.artifact_utils import dataset_identity, file_identity, atomic_json, atomic_npy
from r2_gaussian.utils.geometry_utils import validate_physical_dataset


@dataclass
class PoseInputs:
    dataset: Path
    pose_path: Path
    iteration: int
    scanner: dict
    records: list
    image_ids: list
    angles: np.ndarray
    pose: dict
    dataset_hash: str
    pose_identity: dict


def _vector(value, size, name, positive=False, counts=False):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    if counts and np.any(result != np.floor(result)):
        raise ValueError(f"{name} must contain integer counts")
    return result


def validate_scanner(value):
    if not isinstance(value, dict) or value.get("mode") != "parallel":
        raise ValueError("Offline pose FBP currently requires scanner.mode='parallel'")
    scanner = dict(value)
    for size, names in ((2, ("nDetector", "sDetector")), (3, ("nVoxel", "sVoxel"))):
        for name in names:
            values = _vector(scanner[name], size, f"scanner.{name}", positive=True, counts=name.startswith("n"))
            scanner[name] = values.astype(int).tolist() if name.startswith("n") else values.tolist()
    if max(scanner["nVoxel"]) > 32767:
        raise ValueError("scanner.nVoxel exceeds NIfTI-1 dimensions")
    for name, size in (("offDetector", 2), ("offOrigin", 3)):
        scanner[name] = _vector(scanner.get(name, [0] * size), size, f"scanner.{name}").tolist()
    for name in ("DSO", "DSD", "accuracy"):
        number = float(scanner.get(name, 0.5 if name == "accuracy" else np.nan))
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"scanner.{name} must be positive and finite")
        scanner[name] = number
    for name, extent, count in (("dDetector", "sDetector", "nDetector"), ("dVoxel", "sVoxel", "nVoxel")):
        expected = np.asarray(scanner[extent]) / np.asarray(scanner[count])
        if name in scanner:
            actual = _vector(scanner[name], len(expected), f"scanner.{name}", positive=True)
            if not np.allclose(actual, expected, rtol=1e-6, atol=0):
                raise ValueError(f"scanner.{name} disagrees with size/count")
        scanner[name] = expected.tolist()
    scanner.setdefault("filter", None)
    return scanner


def load_projection(path, shape):
    array = np.load(path, allow_pickle=False)
    if array.shape != tuple(shape) or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"Expected a floating projection with shape {tuple(shape)}: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite projection: {path}")
    with np.errstate(over="ignore"):
        array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"Projection overflows float32: {path}")
    return array


def _scalar(data, name, expected):
    if name not in data or data[name].shape != () or data[name].item() != expected:
        raise ValueError(f"Pose {name} does not match the dataset/iteration or supported format")


def load_inputs(dataset, pose_path, iteration):
    """Use the main pipeline's content identity and binding/version conventions.

    Binding checks deliberately stay CPU-only; importing r2_gaussian.pose
    would eagerly import torch, even for projection-only exports.
    """
    dataset, pose_path = Path(dataset).resolve(), Path(pose_path).resolve()
    if isinstance(iteration, bool) or int(iteration) != iteration or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    metadata = json.loads((dataset / "meta_data.json").read_text(encoding="utf-8"))
    validate_physical_dataset(metadata)
    scanner = validate_scanner(metadata.get("scanner"))
    records = metadata.get("proj_train")
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset must contain nonempty proj_train records")
    records = [dict(record) for record in records]
    image_ids, angles = [], []
    for record in records:
        name = str(record["file_path"]).replace("\\", "/")
        record["file_path"] = name
        # Match dataset_readers.py: basename, then the first dot, not Path.stem.
        image_ids.append(name.rsplit("/", 1)[-1].split(".")[0])
        angles.append(float(record["angle"]))
        load_projection(dataset / name, scanner["nDetector"])
    angles = np.asarray(angles, dtype=np.float64)
    if len(set(image_ids)) != len(image_ids) or not all(image_ids):
        raise ValueError("Training image IDs must be nonempty and unique")
    if not np.isfinite(angles).all():
        raise ValueError("Training angles must be finite radians")
    dataset_hash = dataset_identity(dataset)
    pose_hash = file_identity(pose_path)
    with np.load(pose_path, allow_pickle=False) as data:
        for name, expected in (("binding_version", 1), ("pose_format_version", 2),
                               ("model_iteration", iteration), ("scanner_mode", "parallel"),
                               ("dataset_identity", dataset_hash)):
            _scalar(data, name, expected)
        saved_ids = data["image_ids"]
        if saved_ids.ndim != 1 or saved_ids.dtype.kind not in "US":
            raise ValueError("Pose image IDs must be a string vector")
        saved_ids = saved_ids.tolist()
        if len(set(saved_ids)) != len(saved_ids) or set(saved_ids) != set(image_ids):
            raise ValueError("Pose image IDs do not match training views")
        lookup = {name: i for i, name in enumerate(saved_ids)}
        indices = [lookup[name] for name in image_ids]
        n = len(indices)
        base_angles = np.asarray(data["base_angles"])
        if (base_angles.shape != (n,) or not np.isfinite(base_angles).all()
                or not np.allclose(base_angles[indices], angles, rtol=0, atol=1e-6)):
            raise ValueError("Pose base angles do not match training views")
        if {"delta_translation", "delta_quaternion"}.intersection(data.files):
            raise ValueError("Cone pose parameters cannot be applied to parallel FBP")
        pose = {}
        for name, shape in {"delta_angle": (n,), "view_shift": (n, 2), "global_shift": (2,),
                            "beam_tilt": (2,), "detector_roll": (1,)}.items():
            value = np.asarray(data[name]) if name in data else np.zeros(shape)
            if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"Pose {name} has invalid shape or NaN/Inf")
            value = value.astype(np.float64)
            pose[name] = value[indices] if name in ("delta_angle", "view_shift") else value
    if file_identity(pose_path) != pose_hash or dataset_identity(dataset) != dataset_hash:
        raise ValueError("Inputs changed during validation")
    return PoseInputs(dataset, pose_path, int(iteration), scanner, records, image_ids,
                      angles, pose, dataset_hash, pose_hash)


def correction_affine(shape, pixel_size_vu, shift_uv, roll):
    """Return SciPy [V,U] output-to-input mapping, including physical pitch."""
    pitch = _vector(pixel_size_vu, 2, "pixel_size_vu", positive=True)[::-1]
    shift = _vector(shift_uv, 2, "shift_uv")
    if not np.isfinite(roll):
        raise ValueError("roll must be finite radians")
    cosine, sine = np.cos(roll), np.sin(roll)
    inverse_roll = np.array([[cosine, sine], [-sine, cosine]])
    uv_matrix = np.diag(1 / pitch) @ inverse_roll @ np.diag(pitch)
    center = (np.asarray(shape, dtype=np.float64)[::-1] - 1) / 2
    uv_offset = center + shift - uv_matrix @ center
    return uv_matrix[::-1, ::-1], uv_offset[::-1]


def correct_projection(projection, pixel_size_vu, shift_uv, roll):
    if np.all(np.asarray(shift_uv) == 0) and roll == 0:
        return projection.astype(np.float32, copy=True), np.ones(projection.shape, dtype=bool)
    matrix, offset = correction_affine(projection.shape, pixel_size_vu, shift_uv, roll)
    corrected = affine_transform(projection, matrix, offset=offset, output=np.float32,
                                 order=1, mode="constant", cval=0.0, prefilter=False)
    # With order=1 and constant mode, a sample is supported iff its coordinates
    # lie within the closed original pixel-center domain on both axes.
    supported = affine_transform(np.ones(projection.shape, np.float32), matrix, offset=offset,
                                 order=1, mode="constant", cval=0.0, prefilter=False)
    return corrected, supported >= 1 - 1e-6


def export_projections(inputs, output, *, shifts=True, roll=True, angles=True, beam_tilt_in_geometry=False):
    output = Path(output)
    (output / "corrected_projections").mkdir(parents=True)
    (output / "valid_masks").mkdir()
    corrected_angles = inputs.angles + (inputs.pose["delta_angle"] if angles else 0)
    views = []
    for index, record in enumerate(inputs.records):
        source = inputs.dataset / record["file_path"]
        projection = load_projection(source, inputs.scanner["nDetector"])
        shift = inputs.pose["global_shift"] + inputs.pose["view_shift"][index] if shifts else np.zeros(2)
        rho = float(inputs.pose["detector_roll"][0]) if roll else 0.0
        corrected, mask = correct_projection(projection, inputs.scanner["dDetector"], shift, rho)
        relative = f"corrected_projections/{index:06d}.npy"
        mask_relative = f"valid_masks/{index:06d}.npy"
        atomic_npy(output / relative, corrected)
        atomic_npy(output / mask_relative, mask)
        views.append({"index": index, "image_id": inputs.image_ids[index],
                      "source": str(source), "source_record": record["file_path"],
                      "projection": relative, "mask": mask_relative,
                      "original_angle_rad": float(inputs.angles[index]),
                      "corrected_angle_rad": float(corrected_angles[index]),
                      "applied_shift_uv_px": shift.tolist(), "applied_roll_rad": rho,
                      "valid_fraction": float(mask.mean())})
    atomic_npy(output / "angles_corrected.npy", corrected_angles)
    atomic_json(output / "views.json", {"format_version": 1, "scanner": inputs.scanner,
                "dataset_identity": inputs.dataset_hash, "pose": inputs.pose_identity,
                "model_iteration": inputs.iteration,
                "beam_tilt_rad": inputs.pose["beam_tilt"].tolist(),
                "beam_tilt_application": "3d_geometry" if beam_tilt_in_geometry else "ignored",
                "geometry_file": "geometry_full_pose.npz" if beam_tilt_in_geometry else None,
                "projection_space": "learned_shift_roll_removed",
                "interpolation": {"order": 1, "mode": "constant", "cval": 0.0},
                "views": views})
    return views


def read_exported_projections(output):
    """Reconstruction consumes the files listed in the saved manifest."""
    output = Path(output)
    manifest = json.loads((output / "views.json").read_text(encoding="utf-8"))
    angles = np.load(output / "angles_corrected.npy", allow_pickle=False)
    views = manifest["views"]
    if angles.shape != (len(views),) or not np.isfinite(angles).all():
        raise ValueError("Saved angles do not match the projection manifest")
    projections = []
    for index, view in enumerate(views):
        if view["index"] != index or float(angles[index]) != view["corrected_angle_rad"]:
            raise ValueError("Saved projection order or angles changed")
        projections.append(load_projection(output / view["projection"], manifest["scanner"]["nDetector"]))
        mask = np.load(output / view["mask"], allow_pickle=False)
        if (mask.shape != tuple(manifest["scanner"]["nDetector"]) or mask.dtype != bool
                or float(mask.mean()) != view["valid_fraction"]):
            raise ValueError("Saved valid mask does not match its projection record")
    return np.stack(projections), angles, manifest
