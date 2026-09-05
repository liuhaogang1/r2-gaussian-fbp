"""Find the rotation center directly from the original TIFF projections."""

import argparse
import csv
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fbp_preprocess import build_angles, process_projection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--angle_start", type=float, default=0.0)
    parser.add_argument("--angle_interval", type=float, default=0.5)
    parser.add_argument("--keep_duplicate_endpoint", action="store_true")
    parser.add_argument(
        "--input_type",
        choices=["transmission", "intensity", "line_integral"],
        default="line_integral",
    )
    parser.add_argument("--i0", type=float, default=1.0)
    parser.add_argument("--i0_percentile", type=float, default=99.5)
    parser.add_argument("--zero_policy", choices=["nearest", "clip", "keep"], default="nearest")
    parser.add_argument("--log_eps", type=float, default=1e-6)
    parser.add_argument("--clip_percentile", type=float, default=99.9)
    parser.add_argument("--shift_v", type=int, default=0)
    parser.add_argument("--pixel_subsample", type=int, default=1)
    parser.add_argument("--resize_order", type=int, choices=[0, 1, 3], default=1)
    parser.add_argument("--projection_scale", type=float, default=1.0)
    parser.add_argument("--pixel_size", type=float, default=0.02)
    parser.add_argument("--method", choices=["vo", "scipy"], default="vo")
    parser.add_argument("--init_px", type=float, default=None)
    parser.add_argument("--tol", type=float, default=0.25)
    parser.add_argument("--algorithm", default="scipy")
    parser.add_argument("--search_min_px", type=float, default=-100.0)
    parser.add_argument("--search_max_px", type=float, default=100.0)
    parser.add_argument("--slice_step", type=int, default=8)
    parser.add_argument("--slice_margin", type=int, default=128)
    parser.add_argument("--max_slices", type=int, default=9)
    parser.add_argument(
        "--axis_slope_threshold",
        type=float,
        default=0.01,
        help="Maximum allowed center drift in pixels per detector row",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def configure_threads(threads):
    threads = max(1, min(int(threads), 64))
    max_threads = max(64, int(os.cpu_count() or 64))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["NUMEXPR_MAX_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    return threads


def main(args):
    threads = configure_threads(args.threads)
    try:
        import tomopy
    except ModuleNotFoundError as exc:
        raise SystemExit("TomoPy is not installed in the active environment") from exc

    try:
        import numexpr

        numexpr.set_num_threads(threads)
    except (ImportError, ValueError):
        pass

    input_dir = args.input_dir.resolve()
    if args.config is None:
        config_candidates = sorted(input_dir.glob("*.txt"))
        args.config = config_candidates[0] if len(config_candidates) == 1 else None
    paths = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not paths:
        raise ValueError(f"No TIFF files found in {input_dir}")

    raw = [tifffile.imread(path) for path in paths]
    processed_all = np.stack([process_projection(image, args) for image in raw], axis=0)
    angles = build_angles(args, len(processed_all))
    if len(angles) < 2:
        raise ValueError("At least two projection angles are required")
    projections = processed_all[: len(angles)]

    # VO needs the actual 0/180 pair. TIGRE FBP will use only unique views.
    vo_projections = projections
    if args.method == "vo" and len(processed_all) == len(angles) + 1:
        vo_projections = processed_all
        print("Using the original 180-degree endpoint for find_center_vo.")

    n_slices = projections.shape[1]
    margin = max(0, int(args.slice_margin))
    indices = np.arange(margin, n_slices - margin, max(1, int(args.slice_step)))
    if indices.size == 0:
        raise ValueError("No detector rows remain after slice_margin")
    if args.max_slices > 0 and indices.size > args.max_slices:
        selected = np.linspace(0, indices.size - 1, args.max_slices).round().astype(int)
        indices = indices[selected]
    print(f"Read {len(paths)} TIFFs from {input_dir}")
    print(f"Processed projections: {processed_all.shape}")
    print(f"Using detector rows: {indices.tolist()}")

    init_px = projections.shape[2] / 2.0 if args.init_px is None else args.init_px
    centers = []
    rows = []
    try:
        find_center_parameters = inspect.signature(tomopy.find_center).parameters
    except (TypeError, ValueError):
        find_center_parameters = {}

    for index in indices:
        print(f"Finding center for detector row {int(index)}...", flush=True)
        if args.method == "vo":
            center = tomopy.find_center_vo(
                vo_projections,
                ind=int(index),
                smin=float(args.search_min_px),
                smax=float(args.search_max_px),
                srad=6.0,
                step=float(args.tol),
                ratio=0.5,
                drop=True,
            )
        else:
            kwargs = {
                "ind": int(index),
                "init": float(init_px),
                "tol": float(args.tol),
                "mask": True,
                "ratio": 0.5,
                "sinogram_order": False,
            }
            if "algorithm" in find_center_parameters:
                kwargs["algorithm"] = args.algorithm
            if "verbose" in find_center_parameters:
                kwargs["verbose"] = False
            center = tomopy.find_center(projections, angles, **kwargs)
        center = float(np.asarray(center).reshape(-1)[0])
        centers.append(center)
        rows.append({"slice": int(index), "center_px": center})

    row_values = np.asarray(indices, dtype=np.float64)
    center_values = np.asarray(centers, dtype=np.float64)
    width = float(projections.shape[2])
    row_reference = (float(n_slices) - 1.0) / 2.0
    # A tilted rotation axis appears as a systematic change of the measured
    # horizontal center along detector rows.  Fit that drift and report it;
    # a single offDetector value can only correct the intercept at mid-row.
    slope, center_at_reference = np.polyfit(
        row_values - row_reference, center_values, 1
    )
    fitted_centers = center_at_reference + slope * (row_values - row_reference)
    residuals = center_values - fitted_centers
    center_px = float(center_at_reference)
    median_center_px = float(np.median(center_values))
    p10 = float(np.percentile(center_values, 10))
    p90 = float(np.percentile(center_values, 90))
    spread = p90 - p10
    spread_limit = max(5.0, 0.02 * width)
    axis_drift_px = float(slope * (n_slices - 1.0))
    axis_tilt_deg = float(np.degrees(np.arctan(slope)))
    residual_p90 = float(np.percentile(np.abs(residuals), 90))
    center_in_range = -0.1 * width <= center_px <= 1.1 * width
    spread_ok = spread <= spread_limit
    axis_ok = abs(float(slope)) <= float(args.axis_slope_threshold)
    center_valid = bool(center_in_range and spread_ok and axis_ok)

    center_reference = width / 2.0
    detector_pixel_u = float(width * args.pixel_size * args.pixel_subsample) / width
    offset_px = center_reference - center_px
    offset_u = offset_px * detector_pixel_u

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["slice", "center_px", "fit_center_px", "residual_px"],
        )
        writer.writeheader()
        for row, center, fitted, residual in zip(
            rows, center_values, fitted_centers, residuals
        ):
            writer.writerow(
                {
                    "slice": row["slice"],
                    "center_px": float(center),
                    "fit_center_px": float(fitted),
                    "residual_px": float(residual),
                }
            )

    result = {
        "offDetector": [float(offset_u), 0.0],
        "center_px": center_px,
        "median_center_px": median_center_px,
        "center_reference_px": center_reference,
        "offset_px": float(offset_px),
        "detector_pixel_u": detector_pixel_u,
        "center_p10": p10,
        "center_p90": p90,
        "center_spread_px": spread,
        "center_spread_limit_px": float(spread_limit),
        "center_fit_row_reference_px": row_reference,
        "axis_slope_px_per_row": float(slope),
        "axis_drift_px_across_detector": axis_drift_px,
        "axis_tilt_deg": axis_tilt_deg,
        "axis_residual_p90_px": residual_p90,
        "center_in_range": center_in_range,
        "spread_ok": spread_ok,
        "axis_ok": axis_ok,
        "center_valid": center_valid,
        "n_tiff": len(paths),
        "processed_shape": [int(x) for x in processed_all.shape],
        "input_dir": str(input_dir),
        "input_type": args.input_type,
        "shift_v": args.shift_v,
        "pixel_subsample": args.pixel_subsample,
        "projection_scale": args.projection_scale,
        "pixel_size": args.pixel_size,
        "method": args.method,
        "axis_slope_threshold": args.axis_slope_threshold,
        "per_slice_csv": str(csv_path.resolve()),
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"TomoPy center at detector mid-row: {center_px:.6f} pixels")
    print(f"Median per-row center: {median_center_px:.6f} pixels")
    print(f"Absolute detector offset: {offset_px:+.6f} pixels")
    print(f"Set scanner.offDetector[0] to approximately {offset_u:.8g}")
    print(
        f"Rotation-axis slope: {slope:+.8f} px/row; "
        f"drift across detector: {axis_drift_px:+.6f} px; "
        f"tilt: {axis_tilt_deg:+.6f} deg"
    )
    print(
        f"Center spread: p10={p10:.6f}, p90={p90:.6f}, "
        f"spread={spread:.6f} px"
    )
    if center_valid:
        print("Center/axis check: PASS; the center JSON may be used for FBP.")
    else:
        print(
            "Center/axis check: FAIL; JSON is diagnostic only and must not be "
            "used for FBP with a single offDetector value."
        )
    print(f"Saved center JSON: {args.output}")
    print(f"Saved per-slice CSV: {csv_path}")


if __name__ == "__main__":
    main(parse_args())
