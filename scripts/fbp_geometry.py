"""CPU-only geometry conventions shared by full-view and train-view FBP."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np


COORDINATE_MODES = ("physical", "gaussian")


def parse_svoxel(value):
    """Accept 'auto' or exactly three positive finite XYZ lengths."""
    if isinstance(value, str):
        value = value.split()
    try:
        items = list(value)
    except TypeError as error:
        raise ValueError("sVoxel requires 'auto' or three positive lengths: X Y Z") from error
    if len(items) == 1 and str(items[0]).lower() == "auto":
        return "auto"
    try:
        numbers = [float(item) for item in items]
    except (ValueError, TypeError) as error:
        raise ValueError("sVoxel requires 'auto' or three positive lengths: X Y Z") from error
    if len(numbers) != 3 or not np.all(np.isfinite(numbers)) or min(numbers) <= 0:
        raise ValueError("sVoxel requires 'auto' or three positive finite lengths: X Y Z")
    return numbers


class SVoxelAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, parse_svoxel(values))
        except ValueError as error:
            raise argparse.ArgumentError(self, str(error)) from error


def detector_shape_after_preprocessing(raw_shape, pixel_subsample):
    """Mirror fbp_preprocess's resize and legacy symmetric square crop."""
    if len(raw_shape) != 2 or any(int(n) <= 0 for n in raw_shape):
        raise ValueError(f"Expected a positive 2D projection shape, got {raw_shape}")
    if int(pixel_subsample) != pixel_subsample or pixel_subsample < 1:
        raise ValueError("pixel_subsample must be a positive integer")
    height, width = (int(n) for n in raw_shape)
    if pixel_subsample != 1:
        height, width = max(1, height // pixel_subsample), max(1, width // pixel_subsample)
        if height > width:
            height -= 2 * ((height - width) // 2)
        elif width > height:
            width -= 2 * ((width - height) // 2)
    return [height, width]


def auto_svoxel_from_projection(path, pixel_size, pixel_subsample):
    """Read only a TIFF header and cover the processed detector with a cube."""
    import tifffile

    pixel_size = float(pixel_size)
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("sVoxel auto requires a positive finite pixel_size")
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        raw_shape = list(tif.pages[0].shape)
    shape = detector_shape_after_preprocessing(raw_shape, pixel_subsample)
    effective_pitch = pixel_size * pixel_subsample
    length = max(shape) * effective_pitch
    values = [length] * 3
    return values, {
        "requested": "auto", "method": "processed_detector_max_extent",
        "projection_file": str(path.resolve()),
        "raw_shape": raw_shape, "processed_shape": shape,
        "input_pixel_size": pixel_size, "pixel_subsample": int(pixel_subsample),
        "effective_pixel_size": effective_pitch,
        "formula": "max(processed_shape) * pixel_size * pixel_subsample",
        "resolved_sVoxel": values,
    }


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
