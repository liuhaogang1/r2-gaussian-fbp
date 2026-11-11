"""Shared selection rules for iteration-bound pose calibration snapshots."""
from pathlib import Path


def resolve_iteration(model_path, requested_iteration):
    """Resolve ``-1`` with the same point-cloud iteration directory used at test time."""
    requested_iteration = int(requested_iteration)
    if requested_iteration == -1:
        from r2_gaussian.utils.system_utils import searchForMaxIteration

        folder = Path(model_path) / "point_cloud"
        if not folder.is_dir():
            raise FileNotFoundError(f"Gaussian point-cloud directory not found: {folder}")
        return int(searchForMaxIteration(str(folder)))
    if requested_iteration < 0:
        raise ValueError("Evaluation iteration must be -1 or a non-negative integer")
    return requested_iteration


def resolve_pose_result(model_path, iteration, explicit_pose=None):
    """Return the matching pose file or reject evaluation that would drop pose."""
    model_path = Path(model_path)
    if explicit_pose:
        result = Path(explicit_pose).resolve()
        if not result.is_file():
            raise FileNotFoundError(f"Pose result not found: {result}")
        return result

    pose_root = model_path / "pose_calibration"
    snapshot = pose_root / f"iteration_{int(iteration)}" / "pose_offsets.npz"
    if snapshot.is_file():
        return snapshot.resolve()
    pose_required = ((pose_root / "pose_required.json").exists()
                     or (pose_root / "pose_offsets.npz").exists())
    if not pose_required:
        from r2_gaussian.utils.config_io import load_saved_config

        try:
            pose_required = bool(load_saved_config(model_path).get("pose_enabled", False))
        except FileNotFoundError:
            # Library callers can supply a model without a CLI configuration.
            pass
    if pose_required:
        raise ValueError("No matching pose snapshot for the selected Gaussian iteration")
    return None


def resolve_test_selection(model_path, requested_iteration, explicit_pose=None):
    iteration = resolve_iteration(model_path, requested_iteration)
    return iteration, resolve_pose_result(model_path, iteration, explicit_pose)
