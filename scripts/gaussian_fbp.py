"""Gaussian FBP compatibility. Called only with explicit gaussian mode."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.fbp_geometry import initialization_positions, reconstruction_scanner


def enabled(args):
    return getattr(args, "coordinate_mode", None) == "gaussian"


def build_geometry(args, detector_shape, legacy_builder):
    physical = argparse.Namespace(**vars(args))
    physical.coordinate_mode = "physical"
    geo = legacy_builder(physical, detector_shape)
    cfg = reconstruction_scanner(
        {"mode": args.mode, "sVoxel": args.sVoxel, "sDetector": geo.sDetector.tolist()},
        "gaussian",
    )
    geo.sDetector = np.asarray(cfg["sDetector"], dtype=np.float32)
    geo.dDetector = geo.sDetector / geo.nDetector
    return geo


def reconstruct(args, projections, angles, legacy_builder, reconstruct_volume):
    from scipy.ndimage import zoom

    factor = args.pixel_subsample if args.recon_pixel_subsample is None else args.recon_pixel_subsample
    geometry_args = argparse.Namespace(**vars(args))
    geometry_args.pixel_subsample = factor
    recon_input = projections
    if factor != args.pixel_subsample:
        height, width = projections.shape[1:]
        new_h = max(1, height * args.pixel_subsample // factor)
        new_w = max(1, width * args.pixel_subsample // factor)
        recon_input = zoom(projections, (1, new_h / height, new_w / width), order=args.resize_order).astype(np.float32)
    geometry = build_geometry(geometry_args, recon_input.shape[1:], legacy_builder)
    print(f"FBP coordinate mode: gaussian; detector V/U size: {geometry.sDetector.tolist()}", flush=True)
    return reconstruct_volume(recon_input, angles, geometry_args, geometry), factor, False


def preview_size(size):
    size = np.asarray(size, dtype=np.float64)
    return size * (2.0 / float(size.max()))


def reconstruction_metadata(scanner):
    return {
        "coordinate_mode": "gaussian",
        "preview_coordinate_system": "normalized_scene",
        "detector_size_for_fbp": reconstruction_scanner(scanner, "gaussian")["sDetector"],
    }


def load_train_volume(args, reconstruct_train, expected_shape):
    """Geometry-verified cache exclusive to explicitly enabled Gaussian FBP."""
    dataset = args.dataset.resolve()
    metadata = json.loads((dataset / "meta_data.json").read_text(encoding="utf-8"))
    output = args.fbp_output.resolve() if args.fbp_output is not None else dataset / "vol_train_fbp_gaussian.npy"
    signature = {
        "version": 1, "coordinate_mode": "gaussian",
        "scanner": reconstruction_scanner(metadata["scanner"], "gaussian"),
        "proj_train": metadata["proj_train"],
        "projection_stats": [
            [str(record["file_path"]), path.stat().st_size, path.stat().st_mtime_ns]
            for record in metadata["proj_train"]
            for path in [dataset / str(record["file_path"]).replace("\\", "/")]
        ],
    }
    if getattr(args, "stripe_correction", None) is not None:
        signature["stripe_correction"] = metadata["source"]["stripe_correction"]
    sidecar = output.with_suffix(".geometry.json")
    cached = None
    if sidecar.exists():
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    if output.exists() and cached == signature:
        print(f"Load cached train-view FBP volume: {output}")
        volume = np.asarray(np.load(output), dtype=np.float32)
    else:
        print(f"Reconstruct Gaussian-coordinate train-view FBP from: {dataset}")
        volume = reconstruct_train(dataset, "gaussian")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, volume)
        sidecar.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    shape = tuple(int(n) for n in expected_shape(dataset))
    if volume.ndim != 3 or (shape and volume.shape != shape):
        raise ValueError(f"Gaussian FBP shape {volume.shape} does not match scanner nVoxel {shape}")
    return volume, output


def point_positions(selected, shape, scanner_dir):
    path = Path(scanner_dir) / "meta_data.json"
    scanner = json.loads(path.read_text(encoding="utf-8"))["scanner"] if path.exists() else None
    return initialization_positions(selected, shape, scanner)
