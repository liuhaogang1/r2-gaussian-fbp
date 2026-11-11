"""Bind pose arrays to the Gaussian iteration and the acquisition views."""
import numpy as np

from r2_gaussian.utils.artifact_utils import dataset_identity, identity


def scene_identity(scene):
    if getattr(scene, "source_path", None):
        return dataset_identity(scene.source_path)
    return identity({"scanner": scene.scanner_cfg, "views": sorted(
        (str(camera.image_name), float(camera.angle)) for camera in scene.train_cameras)})


def pose_metadata(scene, iteration):
    names = [str(camera.image_name) for camera in scene.train_cameras]
    if len(names) != len(set(names)):
        raise ValueError("Pose binding requires unique training image identifiers")
    return {"binding_version": np.asarray(1, dtype=np.int32),
            "model_iteration": np.asarray(iteration, dtype=np.int64),
            "dataset_identity": np.asarray(scene_identity(scene)),
            "scanner_mode": np.asarray(scene.scanner_cfg["mode"]),
            "image_ids": np.asarray(names),
            "base_angles": np.asarray([float(c.angle) for c in scene.train_cameras])}


def gaussian_checkpoint_metadata(scene, iteration):
    return {"binding_version": 1, "model_iteration": int(iteration),
            "dataset_identity": scene_identity(scene)}


def validate_gaussian_checkpoint(binding, scene, iteration):
    if binding != gaussian_checkpoint_metadata(scene, iteration):
        raise ValueError("Gaussian checkpoint does not match this dataset, geometry, or iteration")


def _validate_binding(data, scene, iteration):
    required = {"binding_version", "model_iteration", "dataset_identity",
                "scanner_mode", "image_ids", "base_angles"}
    if not required.issubset(data.keys()):
        raise ValueError("Unbound legacy pose file: retrain/export a matching iteration snapshot")
    if (int(data["binding_version"]) != 1
            or int(data["model_iteration"]) != iteration
            or str(data["scanner_mode"]) != scene.scanner_cfg["mode"]
            or str(data["dataset_identity"]) != scene_identity(scene)):
        raise ValueError("Pose file does not match this dataset, scanner, or model iteration")
    names_array = np.asarray(data["image_ids"])
    if names_array.ndim != 1:
        raise ValueError("Pose image identifiers must be a vector")
    names = names_array.tolist()
    current = [str(c.image_name) for c in scene.train_cameras]
    if len(set(names)) != len(names) or len(set(current)) != len(current) or set(names) != set(current):
        raise ValueError("Pose image identifiers do not match training views")
    indices = [names.index(name) for name in current]
    angles = np.asarray(data["base_angles"])
    if angles.shape != (len(names),) or not np.isfinite(angles).all():
        raise ValueError("Pose base angle array has an invalid shape or values")
    if not np.allclose(angles[indices], [float(c.angle) for c in scene.train_cameras],
                       rtol=0, atol=1e-6):
        raise ValueError("Pose base angles do not match training views")
    return indices


def validate_checkpoint_binding(binding, scene, iteration):
    # Optimizer moments and model parameters must use the same view ordering.
    indices = _validate_binding(binding, scene, iteration)
    if indices != list(range(len(indices))):
        raise ValueError("Pose checkpoint view order changed; refusing to reorder optimizer state")


def validate_pose_file(path, scene, iteration):
    """Validate and return an index mapping into the saved per-view arrays."""
    with np.load(path, allow_pickle=False) as data:
        indices = _validate_binding(data, scene, iteration)
        n = len(indices)
        shapes = {"delta_angle": (n,), "view_shift": (n, 2),
                  "global_shift": (2,), "beam_tilt": (2,), "detector_roll": (1,),
                  "delta_translation": (n, 3, 1), "delta_quaternion": (n, 4)}
        for key, shape in shapes.items():
            if key in data and (data[key].shape != shape or not np.isfinite(data[key]).all()):
                raise ValueError(f"Pose {key} has an invalid shape or contains NaN/Inf")
    return indices
