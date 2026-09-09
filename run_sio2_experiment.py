"""Run one SiO2 initialization experiment using the existing train/test code."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["full_init", "sparse_fbp_init"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--init_volume", type=Path, default=None)
    parser.add_argument("--n_points", type=int, default=50000)
    parser.add_argument("--density_thresh", type=float, default=0.05)
    parser.add_argument("--density_rescale", type=float, default=0.15)
    parser.add_argument("--normalize_percentile", type=float, default=99.5)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    return parser.parse_args()


def main(args):
    dataset = args.dataset.resolve()
    model = args.model_path.resolve()
    if args.init_volume is None:
        if args.experiment == "full_init":
            raise ValueError("--init_volume is required for full_init")
        init_volume = dataset / "vol_sparse75_fbp.npy"
    else:
        init_volume = args.init_volume.resolve()

    init_output = dataset / f"init_{args.experiment}.npy"
    run([
        sys.executable, str(ROOT / "init_from_fbp.py"),
        "--volume", str(init_volume), "--output", str(init_output),
        "--n_points", str(args.n_points), "--density_thresh", str(args.density_thresh),
        "--density_rescale", str(args.density_rescale),
        "--normalize_percentile", str(args.normalize_percentile),
        "--seed", str(args.seed), "--visualize", "--visualize_slices",
    ])

    if not args.skip_train:
        run([
            sys.executable, str(ROOT / "train.py"),
            "-s", str(dataset), "-m", str(model), "--ply_path", str(init_output),
            "--iterations", str(args.iterations),
        ])
    if not args.skip_test:
        run([
            sys.executable, str(ROOT / "test.py"),
            "-s", str(dataset), "-m", str(model), "--iteration", "-1",
        ])


if __name__ == "__main__":
    main(parse_args())
