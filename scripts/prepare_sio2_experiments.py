"""Prepare the SiO2 experiments while leaving the legacy FBP pipeline intact.

The script deliberately keeps the project dataset layout (``meta_data.json``,
``proj_train`` and ``proj_test``), but adds flat-field correction and writes
both full-view and 75-view FBP volumes.  Run once with ``--preprocess_only``
to create the corrected projections for rotation-center estimation, then run
again with ``--center_json`` to create the final dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, zoom


def fill_nearest(image: np.ndarray, bad: np.ndarray) -> np.ndarray:
    if not bad.any():
        return image
    if (~bad).sum() == 0:
        raise ValueError("Projection contains no valid pixels")
    _, indices = distance_transform_edt(
        bad, return_distances=True, return_indices=True
    )
    image[bad] = image[tuple(indices[:, bad])]
    return image


def process_corrected(image, flat, args):
    image = np.asarray(image, dtype=np.float32)
    flat = np.asarray(flat, dtype=np.float32)
    dark = float(args.dark_value)
    denominator = flat - dark
    valid_flat = denominator > float(args.flat_eps)
    if not valid_flat.any():
        raise ValueError("Flat field contains no valid pixels")
    denominator = np.maximum(denominator, float(args.flat_eps))
    corrected = (image - dark) / denominator
    corrected = np.nan_to_num(corrected, nan=0.0, posinf=1.0, neginf=0.0)
    bad = (~valid_flat) | (~np.isfinite(corrected)) | (corrected <= 0)
    corrected = fill_nearest(corrected, bad)
    corrected = np.clip(corrected, float(args.log_eps), 1.0)
    projection = -np.log(corrected)

    if args.shift_v:
        shifted = np.zeros_like(projection)
        if args.shift_v > 0:
            shifted[:-args.shift_v] = projection[args.shift_v:]
        else:
            shift = -args.shift_v
            shifted[shift:] = projection[:-shift]
        projection = shifted

    if args.pixel_subsample != 1:
        height, width = projection.shape
        new_height = max(1, height // args.pixel_subsample)
        new_width = max(1, width // args.pixel_subsample)
        projection = zoom(
            projection,
            (new_height / height, new_width / width),
            order=args.resize_order,
        )
        if projection.shape[0] > projection.shape[1]:
            offset = (projection.shape[0] - projection.shape[1]) // 2
            projection = projection[offset: projection.shape[0] - offset]
        elif projection.shape[1] > projection.shape[0]:
            offset = (projection.shape[1] - projection.shape[0]) // 2
            projection = projection[:, offset: projection.shape[1] - offset]

    projection *= np.float32(args.projection_scale)
    if not np.isfinite(projection).all():
        raise ValueError("Processed projection contains NaN or Inf")
    return projection.astype(np.float32)


def angles_from_config(config_path: Path, n_views: int):
    values = {}
    if config_path and config_path.exists():
        for line in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    start = float(values.get("AngleFirst", 0.0))
    interval = float(values.get("AngleInterval", 1.0))
    configured = int(values.get("NumberImages", n_views))
    if configured != n_views:
        raise ValueError(f"config declares {configured} views, found {n_views}")
    degrees = start + np.arange(n_views, dtype=np.float32) * interval
    return np.deg2rad(degrees).astype(np.float32), interval


def load_or_make_corrected(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = args.output_dir / "corrected_all.npy"
    angles_path = args.output_dir / "angles_unique.npy"
    if corrected_path.exists() and angles_path.exists() and not args.reprocess:
        corrected = np.load(corrected_path).astype(np.float32)
        angles = np.load(angles_path).astype(np.float32)
        return corrected, angles

    paths = sorted(args.input_dir.glob(args.projection_glob))
    flat_paths = sorted(args.input_dir.glob(args.flat_glob))
    if not paths:
        raise ValueError(f"No projection TIFF files matching {args.projection_glob}")
    if not flat_paths:
        raise ValueError(f"No flat-field TIFF files matching {args.flat_glob}")
    raw_flat = np.stack([tifffile.imread(path) for path in flat_paths], axis=0)
    flat = np.median(raw_flat.astype(np.float32), axis=0)
    raw_paths = [tifffile.imread(path) for path in paths]
    corrected = np.stack([process_corrected(image, flat, args) for image in raw_paths])
    angles_all, interval = angles_from_config(args.config, len(paths))
    if len(paths) >= 2 and np.isclose(interval * (len(paths) - 1), 180.0, atol=1e-3):
        angles = angles_all[:-1]
    else:
        angles = angles_all
    np.save(corrected_path, corrected)
    np.save(angles_path, angles)
    np.save(args.output_dir / "flat_field_median.npy", flat.astype(np.float32))
    with (args.output_dir / "preprocess_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "projection_files": [path.name for path in paths],
                "flat_files": [path.name for path in flat_paths],
                "raw_shape": [int(x) for x in raw_paths[0].shape],
                "corrected_shape": [int(x) for x in corrected.shape],
                "removed_duplicate_endpoint": len(angles_all) != len(angles),
                "projection_glob": args.projection_glob,
                "flat_glob": args.flat_glob,
                "dark_value": args.dark_value,
                "pixel_subsample": args.pixel_subsample,
                "projection_scale": args.projection_scale,
            },
            handle,
            indent=2,
        )
    return corrected, angles


def build_geometry(args, detector_shape, off_detector):
    import tigre

    height, width = detector_shape
    geo = tigre.geometry(mode="parallel", nVoxel=np.asarray(args.nVoxel[::-1], dtype=np.int32))
    geo.DSD = float(args.DSD)
    geo.DSO = float(args.DSO)
    geo.nDetector = np.asarray([height, width], dtype=np.int32)
    geo.sDetector = np.asarray(
        [height * args.pixel_size * args.pixel_subsample,
         width * args.pixel_size * args.pixel_subsample], dtype=np.float32
    )
    geo.dDetector = geo.sDetector / geo.nDetector
    geo.nVoxel = np.asarray(args.nVoxel[::-1], dtype=np.int32)
    geo.sVoxel = np.asarray(args.sVoxel[::-1], dtype=np.float32)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.asarray(args.offOrigin[::-1], dtype=np.float32)
    geo.offDetector = np.asarray([off_detector[1], off_detector[0]], dtype=np.float32)
    geo.accuracy = float(args.accuracy)
    geo.filter = args.filter
    return geo


def save_split(output, name, indices, projections, angles):
    directory = output / name
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index in indices:
        filename = f"{int(index):04d}.npy"
        np.save(directory / filename, projections[index])
        records.append({"file_path": f"{name}/{filename}", "angle": float(angles[index])})
    return records


def make_dataset(args, corrected, angles):
    import tigre.algorithms as algs

    if len(corrected) == len(angles) + 1:
        projections = corrected[:-1]
    elif len(corrected) == len(angles):
        projections = corrected
    else:
        raise ValueError("Corrected projection count and angle count are inconsistent")
    if len(projections) < 75:
        raise ValueError("At least 75 unique views are required")
    full_indices = np.arange(len(projections), dtype=np.int64)
    train_indices = np.linspace(0, len(projections) - 1, 75).round().astype(np.int64)
    train_indices = np.unique(train_indices)
    if len(train_indices) != 75:
        raise ValueError("The uniform 75-view selection was not unique")
    remaining = sorted(set(full_indices.tolist()) - set(train_indices.tolist()))
    rng = random.Random(args.seed)
    test_indices = np.asarray(sorted(rng.sample(remaining, min(args.n_test, len(remaining)))), dtype=np.int64)
    sparse = projections[train_indices]
    geo = build_geometry(args, projections.shape[1:], args.offDetector)
    volume_full = np.transpose(algs.fbp(projections[:, ::-1, :], geo, angles), (2, 1, 0)).astype(np.float32)
    volume_sparse = np.transpose(algs.fbp(sparse[:, ::-1, :], geo, angles[train_indices]), (2, 1, 0)).astype(np.float32)
    volume_full = np.nan_to_num(volume_full, nan=0.0, posinf=0.0, neginf=0.0)
    volume_sparse = np.nan_to_num(volume_sparse, nan=0.0, posinf=0.0, neginf=0.0)
    volume_full = np.clip(volume_full, 0.0, None)
    volume_sparse = np.clip(volume_sparse, 0.0, None)
    np.save(args.output_dir / "proj_all.npy", projections)
    np.save(args.output_dir / "proj_sparse75.npy", sparse)
    np.save(args.output_dir / "vol_full_fbp.npy", volume_full)
    np.save(args.output_dir / "vol_sparse75_fbp.npy", volume_sparse)
    # Keep the legacy Scene reader unchanged: vol_fbp.npy is the reference volume.
    np.save(args.output_dir / "vol_fbp.npy", volume_full)
    train_records = save_split(args.output_dir, "proj_train", train_indices, projections, angles)
    test_records = save_split(args.output_dir, "proj_test", test_indices, projections, angles)
    step_angle = float(angles[1] - angles[0]) if len(angles) > 1 else 0.0
    total_angle = float(np.rad2deg(angles[-1] - angles[0] + step_angle))
    scanner = {
        "mode": "parallel", "DSD": float(args.DSD), "DSO": float(args.DSO),
        "nDetector": [int(x) for x in projections.shape[1:]],
        "sDetector": [float(projections.shape[1] * args.pixel_size * args.pixel_subsample),
                      float(projections.shape[2] * args.pixel_size * args.pixel_subsample)],
        "nVoxel": [int(x) for x in args.nVoxel], "sVoxel": [float(x) for x in args.sVoxel],
        "offOrigin": [float(x) for x in args.offOrigin],
        "offDetector": [float(x) for x in args.offDetector], "accuracy": float(args.accuracy),
        "totalAngle": total_angle,
        "startAngle": float(np.rad2deg(angles[0])), "noise": False, "filter": args.filter,
    }
    metadata = {
        "scanner": scanner, "vol": "vol_fbp.npy", "radius": 1.0,
        "bbox": [
            (np.asarray(args.offOrigin) - np.asarray(args.sVoxel) / 2).tolist(),
            (np.asarray(args.offOrigin) + np.asarray(args.sVoxel) / 2).tolist(),
        ],
        "proj_train": train_records, "proj_test": test_records,
        "source": {
            "input_dir": str(args.input_dir.resolve()), "algorithm": "TIGRE parallel-beam FBP",
            "n_views": len(angles), "angles_all": [float(x) for x in angles],
            "train_indices": train_indices.tolist(), "test_indices": test_indices.tolist(),
            "sparse_views": 75, "input_type": "flat_corrected_intensity",
            "offDetector": [float(x) for x in args.offDetector],
            "center_json": str(args.center_json.resolve()) if args.center_json else None,
        },
    }
    with (args.output_dir / "meta_data.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (args.output_dir / "split_indices.json").open("w", encoding="utf-8") as handle:
        json.dump({"train_indices": train_indices.tolist(), "test_indices": test_indices.tolist()}, handle, indent=2)
    print(f"Saved full and sparse FBP volumes to {args.output_dir}")
    print(f"Training views: {len(train_indices)}; test views: {len(test_indices)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--center_json", type=Path, default=None)
    parser.add_argument("--preprocess_only", action="store_true")
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--projection_glob", default="Tomo_*.tiff")
    parser.add_argument("--flat_glob", default="Flat_*.tiff")
    parser.add_argument("--dark_value", type=float, default=0.0)
    parser.add_argument("--flat_eps", type=float, default=1e-6)
    parser.add_argument("--log_eps", type=float, default=1e-6)
    parser.add_argument("--shift_v", type=int, default=0)
    parser.add_argument("--pixel_subsample", type=int, default=4)
    parser.add_argument("--resize_order", type=int, choices=[0, 1, 3], default=1)
    parser.add_argument("--projection_scale", type=float, default=0.125)
    parser.add_argument("--pixel_size", type=float, default=0.02)
    parser.add_argument("--nVoxel", nargs=3, type=int, default=[256, 256, 256])
    parser.add_argument("--sVoxel", nargs=3, type=float, default=[2.0, 2.0, 2.0])
    parser.add_argument("--offOrigin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--offDetector", nargs=2, type=float, default=[0.0, 0.0])
    parser.add_argument("--DSD", type=float, default=7.0)
    parser.add_argument("--DSO", type=float, default=5.0)
    parser.add_argument("--accuracy", type=float, default=0.5)
    parser.add_argument("--filter", choices=["ram_lak", "shepp_logan", "cosine", "hamming", "hann"], default="hann")
    parser.add_argument("--n_test", type=int, default=105)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args):
    if args.config is None:
        candidates = sorted(args.input_dir.glob("*.txt"))
        args.config = candidates[0] if len(candidates) == 1 else None
    corrected, angles = load_or_make_corrected(args)
    if args.preprocess_only:
        print(f"Saved corrected projections: {args.output_dir / 'corrected_all.npy'}")
        print(f"Saved unique-view angles: {args.output_dir / 'angles_unique.npy'}")
        return
    if args.center_json is not None:
        with args.center_json.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("center_valid") is False:
            raise ValueError(f"Center result is invalid: {args.center_json}")
        args.offDetector = [float(x) for x in result.get("offDetector", [0.0, 0.0])]
    make_dataset(args, corrected, angles)


if __name__ == "__main__":
    main(parse_args())
