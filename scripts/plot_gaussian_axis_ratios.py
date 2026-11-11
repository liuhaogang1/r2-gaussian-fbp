"""Plot principal-axis ratios from a trusted R2-Gaussian training snapshot."""

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def find_snapshot(model_dir, iteration):
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Training output directory does not exist: {model_dir}")

    point_cloud_dir = model_dir / "point_cloud"
    if iteration is not None:
        snapshot = point_cloud_dir / f"iteration_{iteration}" / "point_cloud.pickle"
        if not snapshot.is_file():
            raise FileNotFoundError(f"Gaussian snapshot not found: {snapshot}")
        return iteration, snapshot

    snapshots = []
    for snapshot in point_cloud_dir.glob("iteration_*/point_cloud.pickle"):
        step = snapshot.parent.name.removeprefix("iteration_")
        if step.isdecimal() and snapshot.is_file():
            snapshots.append((int(step), snapshot))
    if not snapshots:
        raise FileNotFoundError(
            f"No point_cloud/iteration_<N>/point_cloud.pickle found in {model_dir}. "
            "Pass the model output directory used for --model_path."
        )
    return max(snapshots, key=lambda item: item[0])


def load_axis_ratios(snapshot):
    with snapshot.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or not {"scale", "scale_bound"} <= data.keys():
        raise ValueError(f"Missing scale or scale_bound in {snapshot}")
    raw_scales = np.asarray(data["scale"], dtype=np.float64)
    if raw_scales.ndim != 2 or raw_scales.shape[1] != 3 or not len(raw_scales):
        raise ValueError(f"Expected a nonempty array of N x 3 scales in {snapshot}")
    if not np.isfinite(raw_scales).all():
        raise ValueError(f"Non-finite Gaussian scales in {snapshot}")

    # The scale activation is monotone, so sorting its inputs orders the axes.
    ordered = np.sort(raw_scales, axis=1)
    bounds = data["scale_bound"]
    if bounds is None:
        # Unbounded axis lengths are exp(raw_scale). Cancel in log space first.
        short, middle, long = ordered.T
        with np.errstate(over="ignore"):
            long_short = np.exp(long - short)
            long_middle = np.exp(long - middle)
    else:
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all():
            raise ValueError(f"Invalid scale_bound in {snapshot}")
        lower, upper = bounds
        if not 0 <= lower < upper:
            raise ValueError(f"Expected 0 <= scale_min < scale_max in {snapshot}")
        # Match GaussianModel.scaling_activation for bounded scales.
        sigmoid = np.exp(-np.logaddexp(0.0, -ordered))
        short, middle, long = (lower + (upper - lower) * sigmoid).T
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            long_short = long / short
            long_middle = long / middle

    if not np.isfinite(long_short).all() or not np.isfinite(long_middle).all():
        raise ValueError(f"Gaussian axis ratios are non-finite in {snapshot}")
    return long_short, long_middle


def save_histograms(model_dir, iteration, ratios, bins):
    long_short, long_middle = ratios
    upper = max(1.01, float(long_short.max()))
    use_log = upper >= 10
    if use_log:
        edges = np.geomspace(1.0, upper, bins + 1)
    else:
        edges = np.linspace(1.0, upper, bins + 1)
    short_counts = np.histogram(long_short, bins=edges)[0]
    middle_counts = np.histogram(long_middle, bins=edges)[0]

    stem = f"gaussian_axis_ratios_iter{iteration}"
    png_path = model_dir / f"{stem}.png"
    csv_path = model_dir / f"{stem}.csv"
    figure = Figure(figsize=(10, 4), layout="constrained")
    axes = figure.subplots(1, 2, sharex=True, sharey=True)
    for axis, counts, label, color in zip(
        axes, (short_counts, middle_counts), ("Long / Short", "Long / Middle"),
        ("#0072B2", "#009E73"),
    ):
        axis.stairs(counts, edges, fill=True, color=color, alpha=0.85)
        axis.set_xlabel(f"{label}{' (log scale)' if use_log else ''}")
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.25)
    if use_log:
        axes[0].set_xscale("log")
    axes[0].set_xlim(edges[0], edges[-1])
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Gaussian count")
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    figure.suptitle(f"Iteration {iteration:,}  |  {len(long_short):,} Gaussians")
    figure.savefig(png_path, dpi=180)

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["bin_left", "bin_right", "long_short_count", "long_middle_count"])
        writer.writerows(zip(edges[:-1], edges[1:], short_counts, middle_counts))
    return png_path, csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Training output directory (--model_path)")
    parser.add_argument("--iteration", type=int, help="Saved iteration (default: latest snapshot)")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins per histogram (default: 50)")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive")

    model_dir = args.model_dir.resolve()
    try:
        iteration, snapshot = find_snapshot(model_dir, args.iteration)
        ratios = load_axis_ratios(snapshot)
        png_path, csv_path = save_histograms(model_dir, iteration, ratios, args.bins)
    except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Iteration {iteration}: {len(ratios[0]):,} Gaussians")
    print(png_path)
    print(csv_path)


if __name__ == "__main__":
    main()
