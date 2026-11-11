"""Sample aligned reference regions from a fixed XYZ density volume."""
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from r2_gaussian.utils.reference_options import validate_reference_options


def _region_slices(start, counts):
    return tuple(slice(int(i), int(i + n)) for i, n in zip(start, counts))


def _axis_starts(length, size, stride):
    last = length - size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


@dataclass(frozen=True)
class ReferenceRegion:
    start: np.ndarray
    counts: np.ndarray
    center: np.ndarray
    size: np.ndarray
    target: np.ndarray
    sample_ratio: float
    generator: np.random.Generator
    block_score: float = None
    block_probability: float = None


class ReferenceSupervision:
    def __init__(self, options, scanner):
        self.options = options
        self.method = options["reference_method"]
        self.path = Path(options["reference_path"]).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Reference volume not found: {self.path}")
        self.volume = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if not isinstance(self.volume, np.ndarray):
            self.volume.close()
            raise ValueError("reference_path must point to a single NPY density volume")
        self.counts = np.asarray(scanner["nVoxel"], dtype=np.int64)
        if self.volume.shape != tuple(self.counts):
            raise ValueError(f"Reference volume shape {self.volume.shape} does not match "
                             f"scanner nVoxel {tuple(self.counts)}")
        if self.volume.dtype.kind not in "iuf":
            raise ValueError("Reference volume must contain real numeric densities")
        # Check the representation actually used for loss computation.
        for plane in self.volume:
            with np.errstate(over="ignore", invalid="ignore"):
                if not np.isfinite(plane.astype(np.float32, copy=False)).all():
                    raise ValueError("Reference volume contains non-finite float32 densities")
        size = np.asarray(scanner["sVoxel"], dtype=np.float64)
        origin = np.asarray(scanner["offOrigin"], dtype=np.float64)
        self.spacing = size / self.counts
        self.lower = origin - size / 2
        self.axis = "xyz".index(options["reference_slice_axis"])

        if self.method == "block":
            self.block_size = np.asarray(options["reference_block_size"])
            if np.any(self.block_size > self.counts):
                raise ValueError("reference_block_size exceeds scanner nVoxel")
            stride = options["reference_block_stride"]
            if stride is None:
                stride = np.maximum(1, self.block_size // 2)
            axes = [_axis_starts(n, b, d) for n, b, d in zip(
                self.counts, self.block_size, stride)]
            self.block_starts = np.asarray(list(product(*axes)), dtype=np.int64)
            self.block_scores = np.asarray([
                np.maximum(self.volume[_region_slices(start, self.block_size)], 0).mean(
                    dtype=np.float64)
                for start in self.block_starts
            ])
            block_count = len(self.block_starts)
            self.block_probabilities = np.full(block_count, 1.0 / block_count)
            if options["reference_block_sampling"] == "density_mix":
                weights = np.sqrt(self.block_scores)
                total = weights.sum()
                if total > 0:
                    mix = options["reference_block_density_mix"]
                    self.block_probabilities *= 1.0 - mix
                    self.block_probabilities += mix * weights / total

    def sample(self, iteration):
        first = self.options["reference_start_iter"]
        if (iteration < first or iteration > self.options["reference_end_iter"]
                or (iteration - first) % self.options["reference_interval"]):
            return None
        rng = np.random.default_rng(np.random.SeedSequence([
            self.options["reference_seed"], iteration,
        ]))
        score = probability = None
        if self.method == "slice":
            start = np.zeros(3, dtype=np.int64)
            start[self.axis] = rng.integers(self.counts[self.axis])
            counts = self.counts.copy()
            counts[self.axis] = 1
            sample_ratio = self.options["reference_slice_sample_ratio"]
        else:
            index = rng.choice(len(self.block_starts), p=self.block_probabilities)
            start = self.block_starts[index]
            counts = self.block_size
            sample_ratio = 1.0
            score = float(self.block_scores[index])
            probability = float(self.block_probabilities[index])
        return ReferenceRegion(
            start=start, counts=counts,
            center=self.lower + (start + counts / 2) * self.spacing,
            size=counts * self.spacing,
            target=self.volume[_region_slices(start, counts)],
            sample_ratio=sample_ratio, generator=rng,
            block_score=score, block_probability=probability,
        )


def prepare_reference_supervision(opt, scanner):
    options = validate_reference_options(vars(opt))
    if options["reference_enabled"]:
        return ReferenceSupervision(options, scanner)
    return None
