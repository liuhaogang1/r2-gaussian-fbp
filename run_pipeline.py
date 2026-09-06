"""Run the refcorr FBP -> initialization -> training -> test pipeline.

This wrapper keeps the four project stages reproducible while leaving the
stage scripts independently usable.  The center scan is optional when a
previously validated center JSON is available.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def command(args: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(str(item) for item in args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    center_source = parser.add_mutually_exclusive_group()
    center_source.add_argument("--center_json", type=Path, default=None)
    center_source.add_argument(
        "--offDetector",
        nargs=2,
        type=float,
        default=None,
        metavar=("U", "V"),
        help="Use this manual detector offset [u, v] and do not run/use a center scan.",
    )
    parser.add_argument(
        "--skip_center",
        action="store_true",
        help="Use zero detector offset when --center_json is not supplied.",
    )
    parser.add_argument("--center_method", choices=["pair", "vo", "scipy"], default="pair")
    parser.add_argument(
        "--center_pixel_subsample",
        type=int,
        default=1,
        help="Downsampling used only while estimating the rotation center; 1 is recommended.",
    )
    parser.add_argument("--slice_margin", type=int, default=80)
    parser.add_argument("--slice_step", type=int, default=32)
    parser.add_argument("--max_slices", type=int, default=9)
    parser.add_argument("--center_tol", type=float, default=0.25)
    parser.add_argument("--center_search_min_px", type=float, default=-100.0)
    parser.add_argument("--center_search_max_px", type=float, default=100.0)
    parser.add_argument("--input_type", choices=["transmission", "intensity", "line_integral"], default="line_integral")
    parser.add_argument("--angle_interval", type=float, default=0.5)
    parser.add_argument("--shift_v", type=int, default=0)
    parser.add_argument("--pixel_subsample", type=int, default=4)
    parser.add_argument("--resize_order", type=int, choices=[0, 1, 3], default=1)
    parser.add_argument("--projection_scale", type=float, default=0.125)
    parser.add_argument("--pixel_size", type=float, default=0.02)
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--nVoxel", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--sVoxel", nargs=3, type=float, default=[2.0, 2.0, 2.0])
    parser.add_argument("--DSD", type=float, default=7.0)
    parser.add_argument("--DSO", type=float, default=5.0)
    parser.add_argument("--filter", choices=["ram_lak", "shepp_logan", "cosine", "hamming", "hann"], default="hann")
    parser.add_argument("--n_points", type=int, default=50000)
    parser.add_argument("--density_thresh", type=float, default=0.05)
    parser.add_argument("--density_rescale", type=float, default=0.15)
    parser.add_argument("--normalize_percentile", type=float, default=99.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_render_train", action="store_true")
    parser.add_argument("--skip_render_test", action="store_true")
    parser.add_argument("--skip_recon", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    model_path = args.model_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    center_json = args.center_json.resolve() if args.center_json else None
    manual_off_detector = args.offDetector
    if args.center_pixel_subsample < 1:
        raise ValueError("--center_pixel_subsample must be at least 1")
    if args.center_search_min_px >= args.center_search_max_px:
        raise ValueError("--center_search_min_px must be smaller than --center_search_max_px")

    if center_json is None and manual_off_detector is None and not args.skip_center:
        center_json = output_dir.with_name(output_dir.name + "_center.json")
        center_command = [
            sys.executable,
            str(ROOT / "scripts" / "scan_fbp_center_tomopy.py"),
            "--input_dir", str(input_dir),
            "--input_type", args.input_type,
            "--angle_interval", str(args.angle_interval),
            "--shift_v", str(args.shift_v),
            "--pixel_subsample", str(args.center_pixel_subsample),
            "--resize_order", str(args.resize_order),
            "--projection_scale", str(args.projection_scale),
            "--pixel_size", str(args.pixel_size),
            "--method", args.center_method,
            "--slice_margin", str(args.slice_margin),
            "--slice_step", str(args.slice_step),
            "--max_slices", str(args.max_slices),
            "--tol", str(args.center_tol),
            "--pair_min_offset_px", str(args.center_search_min_px),
            "--pair_max_offset_px", str(args.center_search_max_px),
            "--output", str(center_json),
        ]
        if args.config is not None:
            center_command += ["--config", str(args.config.resolve())]
        command(center_command)
    elif manual_off_detector is not None:
        print(
            "Using manual offDetector [u, v]: "
            f"{manual_off_detector[0]:.8g} {manual_off_detector[1]:.8g}",
            flush=True,
        )
    elif center_json is not None:
        print(f"Using existing center JSON: {center_json}", flush=True)
    else:
        print("Center scan skipped; using zero detector offset.", flush=True)

    fbp_command = [
        sys.executable,
        str(ROOT / "prepare_fbp_tiff.py"),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--input_type", args.input_type,
        "--angle_interval", str(args.angle_interval),
        "--shift_v", str(args.shift_v),
        "--pixel_subsample", str(args.pixel_subsample),
        "--resize_order", str(args.resize_order),
        "--projection_scale", str(args.projection_scale),
        "--pixel_size", str(args.pixel_size),
        "--n_train", str(args.n_train),
        "--n_test", str(args.n_test),
        "--nVoxel", *map(str, args.nVoxel),
        "--sVoxel", *map(str, args.sVoxel),
        "--DSD", str(args.DSD),
        "--DSO", str(args.DSO),
        "--filter", args.filter,
        "--seed", str(args.seed),
    ]
    if args.config is not None:
        fbp_command += ["--config", str(args.config.resolve())]
    if center_json is not None:
        fbp_command += ["--center_json", str(center_json)]
    elif manual_off_detector is not None:
        fbp_command += ["--offDetector", *map(str, manual_off_detector)]
    command(fbp_command)

    init_output = output_dir / f"init_{output_dir.name}.npy"
    init_command = [
        sys.executable,
        str(ROOT / "init_from_fbp.py"),
        "--volume", str(output_dir / "vol_fbp.npy"),
        "--output", str(init_output),
        "--n_points", str(args.n_points),
        "--density_thresh", str(args.density_thresh),
        "--density_rescale", str(args.density_rescale),
        "--normalize_percentile", str(args.normalize_percentile),
        "--seed", str(args.seed),
        "--visualize",
        "--visualize_slices",
    ]
    command(init_command)

    if not args.skip_train:
        command([
            sys.executable,
            str(ROOT / "train.py"),
            "-s", str(output_dir),
            "-m", str(model_path),
            "--ply_path", str(init_output),
            "--iterations", str(args.iterations),
        ])

    if not args.skip_test:
        test_command = [
            sys.executable,
            str(ROOT / "test.py"),
            "-s", str(output_dir),
            "-m", str(model_path),
            "--iteration", "-1",
        ]
        if args.skip_render_train:
            test_command.append("--skip_render_train")
        if args.skip_render_test:
            test_command.append("--skip_render_test")
        if args.skip_recon:
            test_command.append("--skip_recon")
        command(test_command)


if __name__ == "__main__":
    main(parse_args())
