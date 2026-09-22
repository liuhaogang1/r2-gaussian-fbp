"""Prepare parallel-beam TIFF projections and reconstruct an FBP volume.

The reconstruction input must be line integrals. For transmission/intensity
TIFFs this script therefore applies ``-log(I / I0)`` before TIGRE FBP.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tifffile
import tigre
import tigre.algorithms as algs
from scripts.fbp_preprocess import (
    build_angles,
    parse_config,
    parse_free_ray,
    process_projection,
)


def save_fbp_preview(output, volume, s_voxel, preview_percentiles=(1.0, 99.5)):
    """Save immediately viewable orthogonal slices and a correctly scaled NIfTI."""
    import SimpleITK as sitk
    from PIL import Image

    volume = np.asarray(volume, dtype=np.float32)
    positive = volume[np.isfinite(volume) & (volume > 0)]
    if positive.size == 0:
        raise ValueError("FBP volume contains no positive voxels for preview")
    low, high = np.percentile(positive, preview_percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(positive.min())
        high = float(positive.max())
    display = np.clip((volume - low) / (high - low), 0.0, 1.0)
    center = tuple(int(size // 2) for size in volume.shape)

    # The array convention is (x, y, z); save all three central planes.
    planes = {
        "x": display[center[0], :, :],
        "y": display[:, center[1], :],
        "z": display[:, :, center[2]],
    }
    for axis, plane in planes.items():
        Image.fromarray(np.round(plane * 255.0).astype(np.uint8)).save(
            output / f"vol_fbp_preview_{axis}.png"
        )

    # Keep the full dynamic range for Slicer; PNGs are only for quick viewing.
    preview_image = sitk.GetImageFromArray(volume.transpose(2, 0, 1))
    spacing = tuple(float(size) / float(count) for size, count in zip(s_voxel, volume.shape))
    preview_image.SetSpacing(spacing)
    sitk.WriteImage(preview_image, str(output / "vol_fbp_preview.nii.gz"))
    return float(low), float(high)


def build_geometry(args, detector_shape):
    """Build the TIGRE geometry for parallel FBP or cone FDK.

    ``args.pixel_size`` is always the *reconstruction-plane* pixel pitch in the
    same length unit as ``DSD``/``DSO``/``sVoxel``.  For a cone-beam scan this
    is the pixel size at the rotation axis, i.e. the detector pixel pitch
    divided by the geometric magnification ``DSD / DSO``.
    """
    height, width = detector_shape
    effective_pixel_size = args.pixel_size * args.pixel_subsample
    mode = args.mode
    if mode not in {"parallel", "cone"}:
        raise ValueError(f"Unsupported scanner mode: {mode}")
    geo = tigre.geometry(mode=mode, nVoxel=np.asarray(args.nVoxel[::-1], dtype=np.int32))
    geo.DSD = float(args.DSD)
    geo.DSO = float(args.DSO)
    geo.nDetector = np.asarray([height, width], dtype=np.int32)
    geo.sDetector = np.asarray(
        [height * effective_pixel_size, width * effective_pixel_size],
        dtype=np.float32,
    )
    geo.dDetector = geo.sDetector / geo.nDetector
    geo.nVoxel = np.asarray(args.nVoxel[::-1], dtype=np.int32)
    geo.sVoxel = np.asarray(args.sVoxel[::-1], dtype=np.float32)
    geo.dVoxel = geo.sVoxel / geo.nVoxel
    geo.offOrigin = np.asarray(args.offOrigin[::-1], dtype=np.float32)
    # The project-wide convention is [u, v]: u is the projection-column
    # direction (the rotation-center direction), and v is the projection-row
    # direction.  Keep this order when passing the offset to TIGRE.  The
    # center scanners write [u, v] and the camera-pose code consumes [u, v] as
    # well; swapping here would make FBP use a different physical offset.
    geo.offDetector = np.asarray(args.offDetector, dtype=np.float32)
    geo.accuracy = float(args.accuracy)
    geo.filter = args.filter
    return geo


def reconstruct_volume(projections, angles, args, geo):
    """FBP for parallel-beam geometry, FDK for cone-beam geometry."""
    if args.mode == "cone":
        volume_tigre = algs.fdk(projections[:, ::-1, :], geo, angles)
    else:
        volume_tigre = algs.fbp(projections[:, ::-1, :], geo, angles)
    return np.transpose(volume_tigre, (2, 1, 0)).astype(np.float32)


def save_split(output, name, indices, projections, angles):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index in indices:
        filename = f"{index:04d}.npy"
        np.save(directory / filename, projections[index])
        records.append(
            {
                "file_path": f"{name}/{filename}",
                "angle": float(angles[index]),
            }
        )
    return records


def reconstruct_with_optional_subsample(args, projections, angles, raw_images):
    """Reconstruct in an optionally coarser geometry than the saved views.

    ``--recon_pixel_subsample`` only affects the analytical reconstruction.
    The saved ``proj_train``/``proj_test`` files keep ``--pixel_subsample`` so
    that changing the reconstruction cost never changes the training data.
    ``raw_images`` must already have the duplicate endpoint removed, i.e. it
    corresponds one-to-one with ``projections`` and ``angles``.
    """
    recon_subsample = (
        args.pixel_subsample
        if args.recon_pixel_subsample is None
        else args.recon_pixel_subsample
    )
    if recon_subsample == args.pixel_subsample:
        geometry_args = args
        reconstruction_input = projections
    else:
        if raw_images is None:
            raise ValueError(
                "--recon_pixel_subsample requires the unprocessed projections"
            )
        if len(raw_images) != len(projections):
            raise ValueError(
                f"Got {len(raw_images)} raw views for {len(projections)} projections"
            )
        geometry_args = argparse.Namespace(**vars(args))
        geometry_args.pixel_subsample = recon_subsample
        reconstruction_input = np.stack(
            [process_projection(image, geometry_args) for image in raw_images],
            axis=0,
        )
        print(
            f"Reconstruction detector {recon_subsample}x subsampled: "
            f"{tuple(reconstruction_input.shape[1:])}",
            flush=True,
        )
    geo = build_geometry(geometry_args, reconstruction_input.shape[1:])
    volume = reconstruct_volume(reconstruction_input, angles, geometry_args, geo)
    return volume, recon_subsample, args.mode == "cone"


def main(args):
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.config = Path(args.config) if args.config else None
    args.center_json = Path(args.center_json) if args.center_json else None
    if args.pixel_subsample < 1:
        raise ValueError("--pixel_subsample must be at least 1")
    if args.recon_pixel_subsample is not None and args.recon_pixel_subsample < 1:
        raise ValueError("--recon_pixel_subsample must be at least 1")
    if args.mode == "cone" and float(args.DSD) <= float(args.DSO):
        raise ValueError("--mode cone requires DSD > DSO")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"FBP output directory is not fresh: {args.output_dir}. "
            "Use a new empty directory for every independent run."
        )
    if args.center_json is not None:
        with args.center_json.open("r", encoding="utf-8") as handle:
            center_result = json.load(handle)
        if "center_valid" in center_result and not center_result["center_valid"]:
            raise ValueError(
                f"Center/axis result is marked invalid: {args.center_json}. "
                "Inspect its axis diagnostics before running FBP."
            )
        off_detector = center_result.get("offDetector")
        if not isinstance(off_detector, list) or len(off_detector) != 2:
            raise ValueError(
                f"Invalid offDetector in center result: {args.center_json}"
            )
        args.offDetector = [float(off_detector[0]), float(off_detector[1])]
        print(
            f"Using offDetector from {args.center_json}: "
            f"{args.offDetector[0]:.8g} {args.offDetector[1]:.8g}"
        )
    if args.config is None:
        config_candidates = sorted(args.input_dir.glob("*.txt"))
        args.config = config_candidates[0] if len(config_candidates) == 1 else None

    paths = sorted(args.input_dir.glob("*.tif"))
    paths += sorted(args.input_dir.glob("*.tiff"))
    if not paths:
        raise ValueError(f"No TIFF files found in {args.input_dir}")

    # --recon_pixel_subsample needs the raw TIFFs again, so keep them only when
    # that option is actually used.
    full_resolution = None
    if args.recon_pixel_subsample is not None:
        full_resolution = [tifffile.imread(path) for path in paths]
        projections_all = np.stack(
            [process_projection(image, args) for image in full_resolution], axis=0
        )
    else:
        projections_all = np.stack(
            [process_projection(tifffile.imread(path), args) for path in paths],
            axis=0,
        )
    angles = build_angles(args, len(projections_all))
    config_values = parse_config(args.config)
    angle_interval = float(config_values.get("AngleInterval", args.angle_interval))
    removed_endpoint = (
        not args.keep_duplicate_endpoint
        and len(paths) == len(angles) + 1
        and np.isclose(angle_interval * len(angles), 180.0, atol=1e-3)
    )
    endpoint_pair = projections_all[[0, -1]].copy() if removed_endpoint else None
    projections = projections_all[: len(angles)]
    if full_resolution is not None:
        # Keep the raw views aligned with 'projections' so that
        # --recon_pixel_subsample cannot reintroduce the dropped endpoint.
        full_resolution = full_resolution[: len(angles)]
    if len(angles) < 2:
        raise ValueError("At least two projection angles are required")
    total_angle = angle_interval * len(angles) if removed_endpoint else float(
        np.rad2deg(angles[-1] - angles[0])
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "proj_all.npy", projections)
    if endpoint_pair is not None:
        np.save(output / "proj_endpoint_pair.npy", endpoint_pair)

    volume_raw, recon_subsample, cone_geometry = reconstruct_with_optional_subsample(
        args, projections, angles, full_resolution
    )
    del full_resolution
    volume_raw = np.nan_to_num(volume_raw, nan=0.0, posinf=0.0, neginf=0.0)
    volume = np.clip(volume_raw, 0.0, None)
    positive_fraction = float(np.count_nonzero(volume) / volume.size)
    if positive_fraction < 0.001:
        # The classic cause is feeding raw transmission counts as line
        # integrals: FBP then returns a nearly flat, mostly negative volume
        # that clips to (almost) nothing.  Fail loudly instead of saving it.
        raise ValueError(
            "The reconstructed volume is essentially empty "
            f"({positive_fraction:.4%} of voxels are positive). "
            f"Check --input_type (currently {args.input_type!r}): transmission "
            "data needs 'transmission', already-log-transformed data needs "
            "'line_integral'. Also check --i0_mode/--i0, --pixel_size and the "
            "reconstruction filter."
        )
    np.save(output / "vol_fbp_raw.npy", volume_raw)
    np.save(output / "vol_fbp.npy", volume)
    preview_low, preview_high = save_fbp_preview(
        output,
        volume,
        args.sVoxel,
    )

    rng = random.Random(args.seed)
    train_indices = np.linspace(0, len(angles) - 1, args.n_train).round().astype(int)
    train_indices = np.unique(train_indices)
    if len(train_indices) != args.n_train:
        raise ValueError("n_train must be smaller than the number of unique views")
    remaining = sorted(set(range(len(angles))) - set(train_indices.tolist()))
    if args.n_test > len(remaining):
        raise ValueError("n_test is larger than the remaining views")
    test_indices = sorted(rng.sample(remaining, args.n_test))

    train_records = save_split(output, "proj_train", train_indices, projections, angles)
    test_records = save_split(output, "proj_test", test_indices, projections, angles)

    scanner = {
        "mode": args.mode,
        "DSD": float(args.DSD),
        "DSO": float(args.DSO),
        "nDetector": [int(x) for x in projections.shape[1:]],
        "sDetector": [
            float(projections.shape[1] * args.pixel_size * args.pixel_subsample),
            float(projections.shape[2] * args.pixel_size * args.pixel_subsample),
        ],
        "nVoxel": [int(x) for x in args.nVoxel],
        "sVoxel": [float(x) for x in args.sVoxel],
        "offOrigin": [float(x) for x in args.offOrigin],
        "offDetector": [float(x) for x in args.offDetector],
        "accuracy": float(args.accuracy),
        "totalAngle": float(total_angle),
        "startAngle": float(np.rad2deg(angles[0])),
        "noise": False,
        "filter": args.filter if args.mode == "parallel" else None,
    }
    metadata = {
        "scanner": scanner,
        "vol": "vol_fbp.npy",
        "radius": 1.0,
        "bbox": [
            (np.asarray(args.offOrigin) - np.asarray(args.sVoxel) / 2).tolist(),
            (np.asarray(args.offOrigin) + np.asarray(args.sVoxel) / 2).tolist(),
        ],
        "proj_train": train_records,
        "proj_test": test_records,
        "source": {
            "input_dir": str(args.input_dir.resolve()),
            "algorithm": (
                "TIGRE cone-beam FDK" if args.mode == "cone" else "TIGRE parallel-beam FBP"
            ),
            "mode": args.mode,
            "n_views": len(angles),
            "angles_all": [float(angle) for angle in angles],
            "removed_duplicate_endpoint": bool(removed_endpoint),
            "input_type": args.input_type,
            "i0": float(args.i0),
            "i0_mode": args.i0_mode,
            "i0_percentile": float(args.i0_percentile),
            "i0_hi_percentile": float(args.i0_hi_percentile),
            "free_ray": [int(x) for x in parse_free_ray(args.free_ray)],
            "zero_policy": args.zero_policy,
            "shift_v": int(args.shift_v),
            "pixel_subsample": int(args.pixel_subsample),
            "recon_pixel_subsample": (
                None if recon_subsample == args.pixel_subsample else int(recon_subsample)
            ),
            "input_pixel_size": float(args.pixel_size),
            "effective_pixel_size": float(args.pixel_size * args.pixel_subsample),
            "recon_pixel_size": float(args.pixel_size * recon_subsample),
            "projection_scale": float(args.projection_scale),
            "endpoint_pair": "proj_endpoint_pair.npy" if endpoint_pair is not None else None,
            "center_json": str(args.center_json.resolve()) if args.center_json else None,
        },
    }
    with (output / "meta_data.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved FBP volume: {output / 'vol_fbp.npy'}")
    print(
        f"Saved FBP previews: {output / 'vol_fbp_preview_z.png'} and "
        f"{output / 'vol_fbp_preview.nii.gz'} "
        f"(window {preview_low:.6g} ~ {preview_high:.6g})"
    )
    print(
        f"Saved {len(angles)} full views, {len(train_records)} training views, "
        f"and {len(test_records)} test views"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--center_json",
        default=None,
        help="Center result JSON from scan_fbp_center_tomopy.py",
    )
    parser.add_argument("--angle_start", type=float, default=0.0)
    parser.add_argument("--angle_interval", type=float, default=0.5)
    parser.add_argument("--keep_duplicate_endpoint", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["parallel", "cone"],
        default="parallel",
        help=(
            "Scanner geometry: parallel uses TIGRE FBP, cone uses TIGRE FDK. "
            "For cone, --DSD/--DSO must be the physical source-detector and "
            "source-origin distances and --pixel_size must be the detector "
            "pitch at the rotation axis (pixel pitch / (DSD/DSO))."
        ),
    )
    parser.add_argument(
        "--input_type",
        choices=["transmission", "intensity", "line_integral"],
        default="transmission",
        help="transmission/intensity applies -log(I/I0); line_integral skips it",
    )
    parser.add_argument(
        "--i0", type=float, default=1.0,
        help="Incident intensity; use 0 to estimate per view",
    )
    parser.add_argument(
        "--i0_mode",
        choices=["percentile", "free_ray", "fixed"],
        default="percentile",
        help=(
            "How I0 is estimated for transmission input: 'percentile' uses "
            "--i0_percentile of the whole frame, 'free_ray' uses the mean of "
            "--free_ray (the HelTomo/FIPS convention), 'fixed' uses --i0."
        ),
    )
    parser.add_argument(
        "--free_ray",
        nargs=4,
        type=int,
        default=[0, 128, 0, 128],
        metavar=("ROW0", "ROW1", "COL0", "COL1"),
        help="Unattenuated detector patch used by --i0_mode free_ray",
    )
    parser.add_argument("--i0_percentile", type=float, default=99.5)
    parser.add_argument(
        "--i0_hi_percentile",
        type=float,
        default=99.9,
        help="Upper clamp for the free-ray I0 estimate",
    )
    parser.add_argument(
        "--zero_policy", choices=["nearest", "clip", "keep"], default="nearest"
    )
    parser.add_argument("--log_eps", type=float, default=1e-6)
    parser.add_argument("--clip_percentile", type=float, default=99.9)
    parser.add_argument(
        "--shift_v", type=int, default=5,
        help="Detector-row shift; 5 matches the old real-data pipeline",
    )
    parser.add_argument("--pixel_subsample", type=int, default=1)
    parser.add_argument(
        "--recon_pixel_subsample",
        type=int,
        default=None,
        help=(
            "Optional detector subsampling used only for the FBP/FDK volume; "
            "defaults to --pixel_subsample. proj_train/proj_test keep "
            "--pixel_subsample."
        ),
    )
    parser.add_argument("--resize_order", type=int, choices=[0, 1, 3], default=1)
    parser.add_argument("--projection_scale", type=float, default=1.0)
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--nVoxel", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--sVoxel", nargs=3, type=float, default=[2.0, 2.0, 2.0])
    parser.add_argument("--offOrigin", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument(
        "--offDetector", nargs=2, type=float, default=[0.0, 0.0],
        help="[u, v] detector offsets in physical units",
    )
    parser.add_argument("--DSD", type=float, default=7.0)
    parser.add_argument("--DSO", type=float, default=5.0)
    parser.add_argument("--pixel_size", type=float, default=1.0)
    parser.add_argument("--accuracy", type=float, default=0.5)
    parser.add_argument(
        "--filter",
        choices=["ram_lak", "shepp_logan", "cosine", "hamming", "hann"],
        default="hann",
    )
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
