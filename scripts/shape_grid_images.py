"""CPU rendering of one XYZ x 25/50/75 percent montage per grid case."""

import json
import math
from pathlib import Path

import numpy as np


FRACTIONS = (0.25, 0.5, 0.75)
PLANES = ((1, 2), (0, 2), (0, 1))  # horizontal/vertical axes in each XYZ plane


def slice_index(length, fraction):
    return min(int(length * fraction), length - 1)


def load_volume(path, expected_shape=None):
    volume = np.load(path, mmap_mode="r", allow_pickle=False)
    if volume.ndim != 3 or any(n < 1 for n in volume.shape):
        raise ValueError(f"Expected a nonempty XYZ volume: {path}")
    if expected_shape is not None and tuple(volume.shape) != tuple(expected_shape):
        raise ValueError(f"Volume grid changed: {path}: {volume.shape} != {expected_shape}")
    for slab in volume:
        if not np.isfinite(slab).all():
            raise ValueError(f"Nonfinite reconstructed density: {path}")
    return volume


def shared_window(cases):
    """Bounded, equally weighted sampling; one window for EVERY case/panel."""
    samples = []
    shape = None
    for case in cases:
        volume = load_volume(case["volume_path"], shape)
        shape = volume.shape
        stride = max(1, math.ceil(volume.size / 200000))
        samples.append(np.array(volume.ravel(order="K")[::stride], dtype=np.float64))
    values = np.concatenate(samples)
    low, high = np.percentile(values, (0.5, 99.5))
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0
    return float(low), float(high), tuple(shape)


def render_case(case, scanner, output_dir, window, expected_shape):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    volume = load_volume(case["volume_path"], expected_shape)
    size = np.asarray(scanner["sVoxel"], dtype=float)
    center = np.asarray(scanner.get("offOrigin", [0, 0, 0]), dtype=float)
    if (size.shape != (3,) or center.shape != (3,) or not np.isfinite(size).all()
            or not np.isfinite(center).all() or np.any(size <= 0)):
        raise ValueError("Invalid scanner XYZ size/center")
    spacing = size / np.asarray(volume.shape)
    lower = center - size / 2
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    slices_dir = output / "slices"
    slices_dir.mkdir(exist_ok=True)
    arrays = {}
    indices = {}
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    try:
        for normal, (horizontal, vertical) in enumerate(PLANES):
            direction = "XYZ"[normal]
            indices[direction] = []
            for column, fraction in enumerate(FRACTIONS):
                index = slice_index(volume.shape[normal], fraction)
                indices[direction].append(index)
                plane = np.take(volume, index, axis=normal)
                arrays[f"{direction.lower()}_{int(fraction * 100)}"] = plane
                axis = axes[normal, column]
                artist = axis.imshow(
                    plane.T, origin="lower", cmap="gray", interpolation="nearest",
                    vmin=window[0], vmax=window[1], aspect="equal",
                    extent=(lower[horizontal], lower[horizontal] + size[horizontal],
                            lower[vertical], lower[vertical] + size[vertical]),
                )
                coordinate = lower[normal] + (index + 0.5) * spacing[normal]
                axis.set_title(
                    f"{direction} = {fraction:.0%} | index {index}\n"
                    f"{direction} coordinate {coordinate:.6g}", fontsize=10,
                )
                axis.set_xlabel(f"{'XYZ'[horizontal]} (dataset units)")
                axis.set_ylabel(f"{'XYZ'[vertical]} (dataset units)")
                axis.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
        fig.colorbar(artist, ax=axes.ravel().tolist(), shrink=0.65, label="Gaussian density")
        if "hard_ratio" in case:
            header = (
                f"{case['case']} | hard action = {case['hard_action']} | iteration {case['iteration']}\n"
                f"soft threshold needle/sheet = {case['needle_threshold']:g}/{case['sheet_threshold']:g} | "
                f"lambda needle/sheet = {case['lambda_needle']:g}/{case['lambda_sheet']:g}\n"
                f"hard trigger needle/sheet = {case['hard_needle_ratio']:g}/{case['hard_sheet_ratio']:g}\n"
            )
        elif "needle_threshold" in case:
            header = (
                f"{case['case']} | shape enabled = {case['shape_enabled']} | hard action = {case['hard_action']}\n"
                f"threshold needle/sheet = {case['needle_threshold']:g}/{case['sheet_threshold']:g} | "
                f"enabled needle/sheet = {case['needle_enabled']}/{case['sheet_enabled']}\n"
                f"lambda needle/sheet = {case['lambda_needle']:g}/{case['lambda_sheet']:g} | iteration {case['iteration']}\n"
            )
        else:
            header = (
                f"{case['case']} | needle/sheet threshold = {case['threshold']:g}\n"
                f"lambda needle = lambda sheet = {case['weight']:g} | iteration {case['iteration']}\n"
            )
        fig.suptitle(
            header +
            f"Rows: X / Y / Z | Columns: 25% / 50% / 75% | shared window [{window[0]:.6g}, {window[1]:.6g}]",
            fontsize=12,
        )
        png_path = output / f"{case['case']}.png"
        fig.savefig(png_path, dpi=180)
    finally:
        plt.close(fig)
    array_path = slices_dir / f"{case['case']}.npz"
    np.savez_compressed(array_path, **arrays)
    return {"case": case["case"], "png": str(png_path), "slices": str(array_path),
            "volume_path": str(case["volume_path"]), "indices_xyz": indices}


def render_grid(cases, metadata_path, output_dir):
    if not cases:
        raise ValueError("No completed final volumes to plot")
    low, high, shape = shared_window(cases)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    scanner = metadata["scanner"]
    if tuple(scanner["nVoxel"]) != shape:
        raise ValueError("Final volume shape does not match shared scanner.nVoxel")
    results = [render_case(case, scanner, output_dir, (low, high), shape) for case in cases]
    report = {"axes": "XYZ", "fractions": list(FRACTIONS), "volume_shape_xyz": list(shape),
              "shared_window": [low, high], "case_count": len(cases),
              "window_method": "pooled equal-stride samples, finite p0.5-p99.5; min/max fallback",
              "orientation": "plane.T, origin=lower; physical XYZ dataset coordinates",
              "cases": results}
    Path(output_dir, "display_settings.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
