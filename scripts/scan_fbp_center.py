"""Estimate absolute horizontal detector offset from a 0/180 pair.

The scan value is an absolute offset relative to the zero-offset geometry.
It is deliberately not added to the current metadata offset, so rerunning
the scan after updating ``offDetector`` does not accumulate the correction.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--u_min_px", type=float, default=-10.0)
    parser.add_argument("--u_max_px", type=float, default=10.0)
    parser.add_argument("--u_step_px", type=float, default=0.5)
    parser.add_argument(
        "--endpoint_pair",
        type=Path,
        default=None,
        help="Two processed views [0 degree, 180 degree] stored as .npy",
    )
    parser.add_argument("--margin_px", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def pair_score(view0, view180, center_px, margin_px):
    height, width = view0.shape
    lo = max(1, int(margin_px))
    hi = min(width - 2, width - int(margin_px) - 1)
    x = np.arange(lo, hi + 1, dtype=np.float32)
    reflected = 2.0 * float(center_px) - x
    valid = (reflected >= 0.0) & (reflected <= width - 1.0)
    x = x[valid]
    reflected = reflected[valid]
    if x.size < 8:
        return -1.0

    source_x = np.arange(width, dtype=np.float32)
    correlations = []
    for row in range(height):
        a = np.interp(x, source_x, view0[row])
        b = np.interp(reflected, source_x, view180[row])
        a -= np.median(a)
        b -= np.median(b)
        denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
        if denom > 1e-10:
            correlations.append(float(np.sum(a * b) / denom))
    return float(np.median(correlations)) if correlations else -1.0


def main(args):
    with (args.data / "meta_data.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    scanner = metadata["scanner"]
    endpoint_path = args.endpoint_pair
    if endpoint_path is None:
        endpoint_name = metadata.get("source", {}).get("endpoint_pair")
        endpoint_path = args.data / endpoint_name if endpoint_name else None
    if endpoint_path is None or not endpoint_path.is_file():
        raise FileNotFoundError(
            "Missing proj_endpoint_pair.npy. Re-run prepare_fbp_tiff.py first."
        )
    endpoint_pair = np.load(endpoint_path).astype(np.float32)
    if endpoint_pair.shape[0] != 2 or endpoint_pair.ndim != 3:
        raise ValueError(f"Expected (2,H,W), got {endpoint_pair.shape}")
    detector_pixel_u = float(scanner["sDetector"][1]) / float(scanner["nDetector"][1])
    candidates = np.arange(
        args.u_min_px, args.u_max_px + args.u_step_px * 0.5, args.u_step_px
    )
    results = []
    best = None

    for u_px in candidates:
        # A positive detector-u offset moves the symmetry axis left in pixels.
        # TIGRE's detector coordinate uses width / 2 as the detector origin.
        symmetry_axis = endpoint_pair.shape[2] / 2.0 - u_px
        score = pair_score(
            endpoint_pair[0], endpoint_pair[1], symmetry_axis, args.margin_px
        )
        result = {
            "u_offset_px": float(u_px),
            "u_offset": float(u_px * detector_pixel_u),
            "pair_score": score,
        }
        results.append(result)
        if best is None or score > best["pair_score"]:
            best = result
        print(f"u_offset_px={u_px:+.2f}, endpoint_pair_score={score:.6g}")

    output = args.output or (args.data / "fbp_center_scan.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["u_offset_px", "u_offset", "pair_score"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Best absolute horizontal detector offset: {best['u_offset_px']:+.3f} pixels")
    print(f"Set scanner.offDetector[0] to approximately {best['u_offset']:.8g}")
    print(f"Saved scan results to {output}")


if __name__ == "__main__":
    main(parse_args())
