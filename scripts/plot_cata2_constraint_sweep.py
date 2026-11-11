"""Plot the 15 final Z-center slices on a common density scale."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


GROUPS = (
    ("slice", "Z-slice reference"),
    ("block", "3D block reference"),
    ("ssim", "Projection SSIM"),
)
WEIGHTS = ("0.1", "0.2", "0.3", "0.4", "0.5")
VOLUME_SHAPE = (256, 256, 256)
Z_INDEX = VOLUME_SHAPE[2] // 2


def load_center_slice(evaluation_dir, iteration):
    status_path = evaluation_dir / "evaluation_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed" or status.get("iteration") != iteration:
        raise ValueError(f"Expected a completed iteration {iteration} evaluation in {evaluation_dir}")

    volume_path = evaluation_dir / "vol_pred.npy"
    volume = np.load(volume_path, mmap_mode="r", allow_pickle=False)
    if volume.shape != VOLUME_SHAPE:
        raise ValueError(f"Expected an XYZ volume of shape {VOLUME_SHAPE} in {volume_path}, got {volume.shape}")
    plane = np.array(volume[:, :, Z_INDEX])
    if not np.isfinite(plane).all():
        raise ValueError(f"Nonfinite values in Z index {Z_INDEX} of {volume_path}")
    return plane


def render_montage(sweep_root, iteration):
    evaluation_dirs = [
        sweep_root / "models" / f"{group}_w{weight}" / "test" / f"iter_{iteration}"
        for group, _ in GROUPS
        for weight in WEIGHTS
    ]
    missing = [str(directory) for directory in evaluation_dirs
               if not all((directory / name).is_file()
                          for name in ("evaluation_status.json", "vol_pred.npy"))]
    if missing:
        raise FileNotFoundError("Missing final evaluation files in:\n" + "\n".join(missing))

    slices = np.stack([load_center_slice(directory, iteration) for directory in evaluation_dirs])
    high = float(np.percentile(slices, 99.5))
    if high <= 0:
        high = max(float(slices.max()), np.finfo(np.float32).eps)
    window = Normalize(vmin=0, vmax=high, clip=True)

    output_path = sweep_root / "figures" / "z_center_15_runs.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(GROUPS), len(WEIGHTS), figsize=(14, 8.5), layout="constrained")
    try:
        for axis, plane in zip(axes.flat, slices):
            image = axis.imshow(plane.T, origin="lower", cmap="gray", norm=window,
                                interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines[:].set_visible(False)

        for axis, weight in zip(axes[0], WEIGHTS):
            axis.set_title(f"Weight {weight}")
        for axis, (_, label) in zip(axes[:, 0], GROUPS):
            axis.set_ylabel(label, rotation=0, labelpad=10, ha="right", va="center")

        fig.suptitle(f"cata2 | Z-center slice (index {Z_INDEX}) | Iteration {iteration:,}")
        fig.colorbar(image, ax=axes, label="Gaussian density", shrink=0.82)
        fig.savefig(output_path, dpi=300)
    finally:
        plt.close(fig)

    print(f"Saved {output_path} (shared density range [0, {high:.6g}])")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()
    try:
        render_montage(args.sweep_root, args.iteration)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
