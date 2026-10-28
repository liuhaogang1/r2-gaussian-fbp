"""Opt-in preprocessing of FBP initialization volumes (never projections)."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_identity(volume_path, metadata_path):
    return {
        "volume_path": str(Path(volume_path).resolve()),
        "volume_sha256": file_digest(volume_path),
        "metadata_path": str(Path(metadata_path).resolve()),
        "metadata_sha256": file_digest(metadata_path),
    }


def validate_settings(sigma, percentile, threshold, density_rescale, n_points):
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("denoise_sigma must be positive and finite")
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError("normalize_percentile must be in [0, 100]")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("density_thresh must be nonnegative and finite")
    if not np.isfinite(density_rescale) or density_rescale <= 0 or n_points < 1:
        raise ValueError("density_rescale and n_points must be positive")


def denoise_volume(volume, sigma):
    # Import SciPy only when this feature is explicitly enabled.
    from scipy.ndimage import gaussian_filter

    start = time.perf_counter()
    result = gaussian_filter(volume, sigma=sigma, mode="nearest", truncate=4.0)
    result = np.maximum(result, 0.0).astype(np.float32, copy=False)
    return result, time.perf_counter() - start


def load_scanner(metadata_path, shape):
    from r2_gaussian.utils.geometry_utils import validate_physical_dataset

    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Denoised initialization requires scanner metadata: {path}. "
            "Keep the source volume in its dataset directory."
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    validate_physical_dataset(metadata)
    scanner = metadata.get("scanner", {})
    size = np.asarray(scanner.get("sVoxel"), dtype=np.float64)
    origin = np.asarray(scanner.get("offOrigin", [0, 0, 0]), dtype=np.float64)
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Denoised initialization requires positive finite scanner.sVoxel")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("scanner.offOrigin must contain three finite coordinates")
    if "nVoxel" in scanner and tuple(scanner["nVoxel"]) != tuple(shape):
        raise ValueError("Initialization volume shape differs from scanner.nVoxel")
    return scanner


def check_output_paths(volume_path, metadata_path, output_dir, point_output):
    """Do not let explicit output paths overwrite the original initializer/input."""
    volume_path = Path(volume_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    output_dir = Path(output_dir).resolve()
    point_output = Path(point_output).resolve()
    if point_output.suffix != ".npy":
        raise ValueError("Denoised initialization point output must have a .npy suffix")
    canonical = volume_path.parent / f"init_{volume_path.parent.name}.npy"
    protected = {volume_path, metadata_path, canonical,
                 metadata_path.parent / f"init_{metadata_path.parent.name}.npy",
                 metadata_path.parent / "vol_fbp.npy",
                 metadata_path.parent / "vol_train_fbp.npy"}
    artifacts = [output_dir / name for name in (
        "vol_init_denoised.npy", "vol_init_denoised.nii.gz", "denoise_report.json",
        *(f"comparison_center_{a}.png" for a in "xyz"),
        *(f"candidate_mask_center_{a}.png" for a in "xyz"))]
    if point_output in protected or any(path in protected for path in artifacts):
        raise ValueError("Denoise output must not overwrite the source volume, metadata or canonical initialization")
    if point_output in artifacts:
        raise ValueError("Initialization point output conflicts with a denoise artifact")


def save_previews(original, denoised, mask, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    # Use exactly the same display window for the original and denoised planes.
    planes = [np.take(original, original.shape[a] // 2, axis=a) for a in range(3)]
    low, high = (float(v) for v in np.percentile(np.concatenate([p.ravel() for p in planes]), [1, 99.5]))
    if high <= low:
        high = low + 1.0
    for axis, name in enumerate("xyz"):
        index = original.shape[axis] // 2
        before = planes[axis].T
        after = np.take(denoised, index, axis=axis).T
        selected = np.take(mask, index, axis=axis).T
        diff = after - before
        limit = max(float(np.max(np.abs(diff))), np.finfo(np.float32).eps)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        try:
            for plot, data, title in zip(axes[:2], (before, after), ("Original FBP", "Denoised FBP")):
                plot.imshow(data, origin="lower", cmap="gray", vmin=low, vmax=high)
                plot.set_title(title)
            delta = axes[2].imshow(diff, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
            axes[2].set_title("Denoised - original")
            fig.colorbar(delta, ax=axes[2], shrink=0.7)
            fig.suptitle(f"{name.upper()} center: index {index} | shared grayscale window")
            fig.savefig(output_dir / f"comparison_center_{name}.png", dpi=140)
        finally:
            plt.close(fig)
        fig, plot = plt.subplots(figsize=(6, 6), constrained_layout=True)
        try:
            plot.imshow(after, origin="lower", cmap="gray", vmin=low, vmax=high)
            overlay = np.zeros((*selected.shape, 4), dtype=np.float32)
            overlay[selected] = (1.0, 0.2, 0.0, 0.4)
            plot.imshow(overlay, origin="lower")
            plot.set_title(f"{name.upper()} center | candidate voxels (before sampling)")
            fig.savefig(output_dir / f"candidate_mask_center_{name}.png", dpi=140)
        finally:
            plt.close(fig)
    return [low, high]


def save_nifti_xyz(volume, scanner, output):
    """Export XYZ voxel centers with dataset lengths and identity LPS axes.

    Unlike the legacy display-only exporter, NIfTI I/J/K are input X/Y/Z.
    No metric unit is asserted: scanner lengths are used without conversion.
    """
    from r2_gaussian.utils.volume_export_utils import save_nifti_streaming

    size = np.asarray(scanner["sVoxel"], dtype=np.float64)
    spacing = size / np.asarray(volume.shape)
    origin = np.asarray(scanner.get("offOrigin", [0, 0, 0])) - size / 2 + spacing / 2
    output = Path(output)
    # The shared streaming writer uses legacy Y/X/Z display axes. Pass a view
    # with X/Y exchanged to produce a true XYZ file without copying the volume.
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, suffix=".nii.gz")
    os.close(descriptor)
    Path(temporary).unlink()
    try:
        save_nifti_streaming(volume.transpose(1, 0, 2), spacing, origin, temporary)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"axes": "XYZ", "spacing": spacing.tolist(), "origin_lps": origin.tolist(),
            "length_units": "dataset units (no conversion)"}


def save_artifacts(original, denoised, mask, selected, point_cloud, scanner,
                   volume_path, metadata_path, output_dir, point_output, args,
                   normalization_scale, elapsed):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    volume_output = output_dir / "vol_init_denoised.npy"
    np.save(volume_output, denoised)
    window = save_previews(original, denoised, mask, output_dir)
    report = {
        "version": 1,
        "input": input_identity(volume_path, metadata_path),
        "algorithm": "scipy.ndimage.gaussian_filter",
        "sigma_voxels": args.denoise_sigma, "mode": "nearest", "truncate": 4.0,
        "shape_xyz": list(denoised.shape), "scanner": scanner,
        "normalization_percentile": args.normalize_percentile,
        "normalization_scale": normalization_scale,
        "density_threshold_normalized": args.density_thresh,
        "density_threshold_before_normalization": args.density_thresh * normalization_scale,
        "density_rescale": args.density_rescale, "seed": args.seed,
        "candidate_count": int(np.count_nonzero(mask)),
        "sampled_count": len(selected), "density_range": [float(point_cloud[:, 3].min()), float(point_cloud[:, 3].max())],
        "filter_seconds": elapsed, "display_window": window,
        "point_output": str(Path(point_output).resolve()),
        "point_sha256": file_digest(point_output),
        "volume_output": str(volume_output), "volume_sha256": file_digest(volume_output),
    }
    if args.denoise_save_nii:
        report["nifti"] = save_nifti_xyz(denoised, scanner, output_dir / "vol_init_denoised.nii.gz")
    report_path = output_dir / "denoise_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Denoise sigma={args.denoise_sigma} voxel, {elapsed:.3f}s; "
          f"normalization={normalization_scale:.6g}; candidates={report['candidate_count']:,}")
    print(f"Saved denoised initialization report: {report_path}")


def pipeline_identity(args):
    """Options which must agree before retaining any denoised-init stage."""
    keys = ("output_dir", "init_source", "init_volume", "init_fbp_output", "init_output",
            "init_denoise_sigma", "init_denoise_output_dir", "init_denoise_save_nii",
            "init_n_points", "init_density_thresh", "init_density_rescale",
            "init_normalize_percentile", "init_seed", "train_source_path",
            "train_model_path", "model_path", "train_ply_path")
    result = {}
    for key in keys:
        value = getattr(args, key, None)
        if isinstance(value, Path):
            value = str(value.resolve())
        result[key] = value
    return result


def validate_resume(args, previous):
    previous_args = previous.get("resolved_arguments", {})
    enabled = bool(getattr(args, "init_denoise", False))
    previous_enabled = bool(previous_args.get("init_denoise", False))
    if enabled != previous_enabled:
        raise ValueError("Initialization denoising changed; use a new model directory without --resume")
    if not enabled:
        return
    if previous.get("denoise_identity") != pipeline_identity(args):
        raise ValueError("Denoised initialization settings changed; use a new model directory without --resume")
    if previous.get("stage_status", {}).get("init") not in {"success", "resumed"}:
        return
    report_path = Path(args.init_denoise_output_dir) / "denoise_report.json"
    if not report_path.exists():
        raise ValueError("Missing denoised initialization report; cannot resume")
    if file_digest(report_path) != previous.get("denoise_report_sha256"):
        raise ValueError("Denoised initialization report changed; cannot resume")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    identity = report["input"]
    if input_identity(identity["volume_path"], identity["metadata_path"]) != identity:
        raise ValueError("FBP input or metadata changed; cannot resume denoised initialization")
    if str(Path(args.init_output).resolve()) != report["point_output"]:
        raise ValueError("Denoised initialization point path changed; cannot resume")
    for path_key, hash_key in (("point_output", "point_sha256"), ("volume_output", "volume_sha256")):
        if file_digest(report[path_key]) != report[hash_key]:
            raise ValueError("Denoised initialization artifact changed; cannot resume")
