"""CPU-only geometry conventions shared by full-view and train-view FBP."""
from copy import deepcopy

import numpy as np


COORDINATE_MODES = ("physical", "gaussian")


def reconstruction_scanner(scanner, coordinate_mode="physical"):
    """Return a reconstruction-only scanner without changing training metadata.

    The current parallel Gaussian renderer maps one normalized scene unit
    directly to one NDC unit. Since scene_scale=2/max(sVoxel), its detector
    spans max(sVoxel) on *both* axes, regardless of the measured pixel pitch.
    Reconstruct with that virtual detector to initialize the existing model.
    This is a compatibility coordinate system, not a physical calibration.
    """
    if coordinate_mode not in COORDINATE_MODES:
        raise ValueError(f"Unknown FBP coordinate mode: {coordinate_mode}")
    result = deepcopy(scanner)
    if coordinate_mode == "physical":
        return result
    if scanner.get("mode") != "parallel":
        raise ValueError("gaussian FBP coordinate mode requires parallel geometry")
    size = np.asarray(scanner["sVoxel"], dtype=np.float64)
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError("sVoxel must contain three positive finite lengths")
    length = float(size.max())
    result["sDetector"] = [length, length]
    if "dDetector" in result:
        result["dDetector"] = (
            np.asarray(result["sDetector"]) / np.asarray(result["nDetector"])
        ).tolist()
    return result


def metadata_coordinate_mode(metadata, override=None):
    mode = override if override is not None else metadata.get("reconstruction", {}).get("coordinate_mode", "physical")
    if mode not in COORDINATE_MODES:
        raise ValueError(f"Unknown FBP coordinate mode: {mode}")
    return mode


def train_fbp_filename(coordinate_mode):
    return "vol_train_fbp_gaussian.npy" if coordinate_mode == "gaussian" else "vol_train_fbp.npy"


def initialization_positions(indices, shape, scanner=None):
    """Map XYZ voxel centers to the same scene coordinates as dataset_readers."""
    fractions = (np.asarray(indices, dtype=np.float32) + 0.5) / np.asarray(shape, dtype=np.float32)
    if scanner is None:
        return fractions * 2.0 - 1.0
    size = np.asarray(scanner["sVoxel"], dtype=np.float32)
    origin = np.asarray(scanner["offOrigin"], dtype=np.float32)
    return (origin + (fractions - 0.5) * size) * (2.0 / float(size.max()))
