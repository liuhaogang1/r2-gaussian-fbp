"""Create an R2-Gaussian initialization point cloud from an FBP volume."""

import argparse
import json
from pathlib import Path

import numpy as np


def reconstruct_from_train_views(dataset: Path) -> np.ndarray:
    """Reconstruct an initialization volume from the dataset training views."""
    metadata_path = dataset / "meta_data.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Cannot use --use_train_fbp: missing {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    scanner_cfg = metadata.get("scanner")
    train_records = metadata.get("proj_train")
    if not isinstance(scanner_cfg, dict):
        raise ValueError(f"Invalid or missing scanner configuration in {metadata_path}")
    if not train_records:
        raise ValueError(f"Dataset has no proj_train records: {metadata_path}")
    if scanner_cfg.get("mode") != "parallel":
        raise ValueError(
            "--use_train_fbp currently supports only parallel-beam datasets"
        )

    projections = []
    angles = []
    projection_shape = None
    for record in train_records:
        if "file_path" not in record or "angle" not in record:
            raise ValueError("Every proj_train record must contain file_path and angle")
        relative_path = str(record["file_path"]).replace("\\", "/")
        projection_path = dataset / relative_path
        if not projection_path.exists():
            raise FileNotFoundError(f"Missing training projection: {projection_path}")
        projection = np.asarray(np.load(projection_path), dtype=np.float32)
        if projection.ndim != 2:
            raise ValueError(
                f"Expected a 2D projection, got {projection.shape} from {projection_path}"
            )
        if projection_shape is None:
            projection_shape = projection.shape
        elif projection.shape != projection_shape:
            raise ValueError(
                "Training projections have inconsistent shapes: "
                f"{projection_shape} and {projection.shape}"
            )
        projections.append(projection)
        angles.append(float(record["angle"]))

    # Import TIGRE only for the train-view FBP mode. Explicit-volume mode keeps
    # the previous behavior and does not need to reconstruct a volume here.
    from r2_gaussian.utils.ct_utils import get_geometry_tigre, recon_volume

    projections_array = np.stack(projections, axis=0)
    angles_array = np.asarray(angles, dtype=np.float32)
    geometry = get_geometry_tigre(scanner_cfg)
    volume = recon_volume(projections_array, angles_array, geometry, "fbp")
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    return volume


def load_initial_volume(args: argparse.Namespace) -> tuple[np.ndarray, Path]:
    """Load an explicit volume or reconstruct/cache one from training views."""
    if args.use_train_fbp:
        if args.dataset is None:
            raise ValueError("--dataset is required when --use_train_fbp is set")
        if args.volume is not None:
            raise ValueError("Do not combine --volume with --use_train_fbp")

        dataset = args.dataset.resolve()
        fbp_output = (
            args.fbp_output.resolve()
            if args.fbp_output is not None
            else dataset / "vol_train_fbp.npy"
        )
        if fbp_output.exists():
            print(f"Load cached train-view FBP volume: {fbp_output}")
            volume = np.asarray(np.load(fbp_output), dtype=np.float32)
        else:
            print(f"Reconstruct train-view FBP volume from: {dataset}")
            volume = reconstruct_from_train_views(dataset)
            fbp_output.parent.mkdir(parents=True, exist_ok=True)
            np.save(fbp_output, volume)
            print(f"Saved train-view FBP cache to {fbp_output}")
        if volume.ndim != 3:
            raise ValueError(f"Expected a 3D FBP volume, got {volume.shape}")
        expected_shape = tuple(int(value) for value in dataset_shape_from_metadata(dataset))
        if expected_shape and volume.shape != expected_shape:
            raise ValueError(
                f"Cached train-view FBP shape {volume.shape} does not match "
                f"scanner nVoxel {expected_shape}: {fbp_output}"
            )
        return volume, fbp_output

    if args.dataset is not None:
        raise ValueError("--dataset requires --use_train_fbp")
    if args.fbp_output is not None:
        raise ValueError("--fbp_output requires --use_train_fbp")
    if args.volume is None:
        raise ValueError(
            "--volume is required unless --use_train_fbp and --dataset are provided"
        )
    return np.asarray(np.load(args.volume), dtype=np.float32), args.volume.resolve()


def dataset_shape_from_metadata(dataset: Path) -> tuple[int, ...]:
    """Read the expected volume shape without importing the reconstruction stack."""
    metadata_path = dataset / "meta_data.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    n_voxel = metadata.get("scanner", {}).get("nVoxel")
    if not n_voxel:
        return ()
    return tuple(int(value) for value in n_voxel)


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
    parser.add_argument(
        "--volume",
        type=Path,
        default=None,
        help="Explicit 3D reconstruction volume (.npy). Required unless --use_train_fbp is set.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Prepared dataset directory containing meta_data.json and proj_train.",
    )
    parser.add_argument(
        "--use_train_fbp",
        action="store_true",
        help="Reconstruct the initialization volume from the dataset training views.",
    )
    parser.add_argument(
        "--fbp_output",
        type=Path,
        default=None,
        help="Optional cached train-view FBP path; defaults to <dataset>/vol_train_fbp.npy.",
    )
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

    volume, volume_path = load_initial_volume(args)
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

    canonical_output = volume_path.parent / f"init_{volume_path.parent.name}.npy"
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
