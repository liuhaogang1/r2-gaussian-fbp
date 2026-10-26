"""Optional per-dataset FOV sizing, enabled only by sVoxel auto."""
import json
from pathlib import Path
import numpy as np


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


def resolve_pipeline(args, stage_in_window):
    if args.fbp_sVoxel != "auto":
        return
    args.fbp_sVoxel_requested = "auto"
    metadata_path = args.output_dir / "meta_data.json"
    if not stage_in_window(args, "fbp") and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = parse_svoxel(metadata["scanner"]["sVoxel"])
        if values == "auto":
            raise ValueError("Existing dataset scanner.sVoxel must contain resolved lengths")
        info = {"requested": "auto", "method": "existing_dataset",
                "metadata_file": str(metadata_path), "resolved_sVoxel": values}
    else:
        from scripts.fbp_preprocess import select_tiff_paths
        pattern = args.fbp_projection_glob
        if args.fbp_flat_field and pattern is None:
            pattern = "Tomo_*.tiff"
        paths = select_tiff_paths(args.input_dir, pattern)
        if not paths:
            raise ValueError(f"sVoxel auto: no projection TIFFs found in {args.input_dir}")
        values, info = auto_svoxel_from_projection(paths[0], args.fbp_pixel_size, args.fbp_pixel_subsample)
    args.fbp_sVoxel = values
    args.fbp_sVoxel_resolution = info
    print(f"Resolved fbp_sVoxel auto: {values} ({info['method']})", flush=True)
