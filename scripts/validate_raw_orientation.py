"""Enumerate RAW axis/flips/inversion and select the best match to a reference volume."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


def normalize(volume):
    volume = np.nan_to_num(np.asarray(volume, dtype=np.float32))
    low, high = np.percentile(volume, [1.0, 99.5])
    if high <= low:
        return np.zeros_like(volume)
    return np.clip((volume - low) / (high - low), 0.0, 1.0)


def resize_volume(volume, shape):
    factors = [float(dst) / float(src) for dst, src in zip(shape, volume.shape)]
    return zoom(volume, factors, order=1).astype(np.float32)


def score(candidate, reference):
    a = normalize(candidate).ravel()
    b = normalize(reference).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2 or np.std(a[valid]) < 1e-8 or np.std(b[valid]) < 1e-8:
        return -1.0
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def save_slices(volume, prefix):
    import matplotlib.pyplot as plt

    volume = normalize(volume)
    center = [int(size // 2) for size in volume.shape]
    plt.imsave(str(prefix.with_name(prefix.name + "_x.png")), volume[center[0], :, :].T, cmap="gray")
    plt.imsave(str(prefix.with_name(prefix.name + "_y.png")), volume[:, center[1], :].T, cmap="gray")
    plt.imsave(str(prefix.with_name(prefix.name + "_z.png")), volume[:, :, center[2]].T, cmap="gray")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--raw_shape", nargs=3, type=int, default=[976, 1024, 1024])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target_shape", nargs=3, type=int, default=None)
    parser.add_argument(
        "--score_shape", nargs=3, type=int, default=[64, 64, 64],
        help="Small volume used for candidate scoring; the selected output uses target_shape.",
    )
    parser.add_argument("--dtype", choices=["uint8", "uint16", "float32"], default="uint8")
    return parser.parse_args()


def main(args):
    dtype = np.dtype(args.dtype)
    raw = np.fromfile(args.raw, dtype=dtype)
    expected = int(np.prod(args.raw_shape))
    if raw.size != expected:
        raise ValueError(f"RAW has {raw.size} values, expected {expected}")
    raw = raw.reshape(tuple(args.raw_shape))
    reference = np.load(args.reference).astype(np.float32)
    target_shape = tuple(args.target_shape or reference.shape)
    score_shape = tuple(args.score_shape)
    reference_score = resize_volume(reference, score_shape) if reference.shape != score_shape else reference
    # Candidate scoring is intentionally performed on a small volume. This
    # avoids resampling the 1 GB RAW file 96 times; the selected orientation
    # is still reconstructed from the full-resolution RAW exactly once.
    raw_score = resize_volume(raw, score_shape)

    results = []
    best = None
    for perm in itertools.permutations(range(3)):
        spatial = np.transpose(raw_score, perm)
        for flip_flags in itertools.product([False, True], repeat=3):
            candidate = spatial
            for axis, should_flip in enumerate(flip_flags):
                if should_flip:
                    candidate = np.flip(candidate, axis=axis)
            for invert in [False, True]:
                volume = candidate.astype(np.float32)
                if invert:
                    volume = float(volume.max()) - volume
                resized = resize_volume(volume, score_shape)
                current = {
                    "perm": list(perm),
                    "flip_axes": [bool(x) for x in flip_flags],
                    "invert": bool(invert),
                    "score": score(resized, reference_score),
                }
                results.append(current)
                if best is None or current["score"] > best["score"]:
                    best = current

    results.sort(key=lambda item: item["score"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_name(args.output.stem + "_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    spatial = np.transpose(raw, tuple(best["perm"]))
    for axis, should_flip in enumerate(best["flip_axes"]):
        if should_flip:
            spatial = np.flip(spatial, axis=axis)
    selected = spatial.astype(np.float32)
    if best["invert"]:
        selected = float(selected.max()) - selected
    selected = resize_volume(selected, target_shape)
    selected = normalize(selected)
    np.save(args.output, selected.astype(np.float32))
    orientation = {
        "raw": str(args.raw.resolve()),
        "raw_shape": list(args.raw_shape),
        "raw_dtype": args.dtype,
        "reference": str(args.reference.resolve()),
        "target_shape": list(target_shape),
        "score_shape": list(score_shape),
        **best,
    }
    with args.output.with_name(args.output.stem + "_orientation.json").open("w", encoding="utf-8") as handle:
        json.dump(orientation, handle, indent=2)
    save_slices(selected, args.output.with_suffix(""))
    print(json.dumps(orientation, indent=2))
    print(f"Saved selected volume: {args.output}")


if __name__ == "__main__":
    main(parse_args())
