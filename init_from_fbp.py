"""Create an R2-Gaussian initialization point cloud from an FBP volume."""

import argparse
from pathlib import Path

import numpy as np


def save_initialization_slices(
    volume_shape: tuple[int, int, int],
    selected: np.ndarray,
    densities: np.ndarray,
    output: Path,
    positions=(0.25, 0.5, 0.75),
) -> None:
    """Save X/Y/Z slices of the selected initialization voxels."""
    import matplotlib.pyplot as plt

    # Reconstruct only the voxels that survive thresholding and random sampling.
    volume = np.zeros(volume_shape, dtype=np.float32)
    volume[tuple(np.asarray(selected, dtype=np.int64).T)] = np.asarray(
        densities, dtype=np.float32
    )
    positive = volume[volume > 0]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    vmax = max(vmax, 1e-6)
    axes = (0, 1, 2)
    axis_names = ("X", "Y", "Z")
    indices = [
        [round(position * (volume.shape[axis] - 1)) for position in positions]
        for axis in axes
    ]

    figure, panels = plt.subplots(3, len(positions), figsize=(4 * len(positions), 10))
    if len(positions) == 1:
        panels = panels[:, None]
    figure.suptitle("FBP initialization volume slices", fontsize=15)
    for row, (axis, axis_name) in enumerate(zip(axes, axis_names)):
        for col, index in enumerate(indices[row]):
            panel = panels[row, col]
            panel.imshow(
                np.take(volume, index, axis=axis).T,
                cmap="gray",
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            panel.set_title(f"{axis_name} {index}", fontsize=10)
            panel.set_xticks([])
            panel.set_yticks([])
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved initialization slices to {output}")


def visualize_initial_point_cloud(
    point_cloud: np.ndarray,
    output: Path,
    max_points: int,
    seed: int,
    show: bool = True,
) -> None:
    """Show and save a 3D scatter preview of the FBP initialization points."""
    import matplotlib.pyplot as plt

    if max_points > 0 and len(point_cloud) > max_points:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(point_cloud), max_points, replace=False))
        point_cloud = point_cloud[selected]

    positions = point_cloud[:, :3]
    densities = point_cloud[:, 3]
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=densities,
        cmap="viridis",
        s=4,
        alpha=0.75,
        linewidths=0,
    )
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title(
        f"FBP initialization ({len(point_cloud):,} points)\n"
        f"density: {densities.min():.4g} - {densities.max():.4g}"
    )
    axis.set_box_aspect(np.maximum(np.ptp(positions, axis=0), 1e-6))
    figure.colorbar(scatter, ax=axis, pad=0.1, label="density")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    print(f"Saved initialization visualization to {output}")
    if show:
        plt.show()
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output point cloud. Defaults to <volume_dir>/init_<volume_dir_name>.npy.",
    )
    parser.add_argument("--n_points", type=int, default=50000)
    parser.add_argument("--density_thresh", type=float, default=0.05)
    parser.add_argument("--density_rescale", type=float, default=0.15)
    parser.add_argument(
        "--normalize_percentile",
        type=float,
        default=99.5,
        help="Normalize positive volume values by this percentile; 0 disables it.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save a 3D scatter preview of the initialized FBP point cloud.",
    )
    parser.add_argument(
        "--visualize_show",
        action="store_true",
        help="Open the preview window after saving it (for desktop use only).",
    )
    parser.add_argument(
        "--visualize_output",
        type=Path,
        default=None,
        help="Preview image path. Defaults to <output_stem>_preview.png.",
    )
    parser.add_argument(
        "--visualize_max_points",
        type=int,
        default=50000,
        help="Maximum points in the preview; 0 keeps all points.",
    )
    parser.add_argument(
        "--visualize_slices",
        action="store_true",
        help="Save slices of the FBP volume used for initialization.",
    )
    parser.add_argument(
        "--visualize_slice_output",
        type=Path,
        default=None,
        help="Slice image path. Defaults to <output_stem>_slices.png.",
    )
    args = parser.parse_args()

    volume = np.load(args.volume).astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape}")
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    volume = np.clip(volume, 0.0, None)

    if args.normalize_percentile > 0:
        positive = volume[volume > 0]
        if positive.size == 0:
            raise ValueError("Volume has no positive voxel")
        scale = float(np.percentile(positive, args.normalize_percentile))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("Invalid volume normalization scale")
        volume = np.clip(volume / scale, 0.0, 1.0)

    mask = volume > args.density_thresh
    indices = np.argwhere(mask)
    if len(indices) < args.n_points:
        raise ValueError(
            f"Only {len(indices)} voxels exceed density_thresh={args.density_thresh}; "
            "lower the threshold or reduce n_points."
        )

    rng = np.random.default_rng(args.seed)
    selected = indices[rng.choice(len(indices), args.n_points, replace=False)]
    shape = np.asarray(volume.shape, dtype=np.float32)
    # Volume coordinates are already in the R2-Gaussian [-1, 1]^3 convention.
    positions = (selected.astype(np.float32) + 0.5) / shape * 2.0 - 1.0
    densities = volume[tuple(selected.T)] * args.density_rescale
    point_cloud = np.concatenate([positions, densities[:, None]], axis=1).astype(np.float32)

    canonical_output = args.volume.parent / f"init_{args.volume.parent.name}.npy"
    output = args.output if args.output is not None else canonical_output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, point_cloud)
    print(f"Saved {point_cloud.shape[0]} points to {output}")

    # initialize_gaussian() looks for init_<dataset_name>.npy when --ply_path
    # is omitted. Keep that convention even when a custom output name is used.
    if output.resolve() != canonical_output.resolve():
        np.save(canonical_output, point_cloud)
        print(f"Saved canonical initialization to {canonical_output}")
    print(f"density range: {densities.min():.6g} ~ {densities.max():.6g}")

    if args.visualize:
        preview_output = args.visualize_output
        if preview_output is None:
            preview_output = output.with_name(output.stem + "_preview.png")
        visualize_initial_point_cloud(
            point_cloud,
            preview_output,
            args.visualize_max_points,
            args.seed,
            show=args.visualize_show,
        )

    if args.visualize_slices:
        slice_output = args.visualize_slice_output
        if slice_output is None:
            slice_output = output.with_name(output.stem + "_slices.png")
        save_initialization_slices(volume.shape, selected, densities, slice_output)

if __name__ == "__main__":
    main()
