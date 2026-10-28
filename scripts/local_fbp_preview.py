"""Create local FBP previews for initialization experiments.

The script writes two independent volumes:
1. a lightly Gaussian-smoothed FBP volume;
2. a screened FBP volume using a normalized density threshold and connected
   component filtering.

The NIfTI writer is deliberately dependency-free so this can run without
SimpleITK or nibabel.
"""

from __future__ import annotations

import argparse
import gzip
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, label


def write_nifti(path: Path, volume: np.ndarray, spacing: tuple[float, float, float]) -> None:
    # NIfTI stores x as the fastest-changing axis.  The project volumes are
    # [X, Y, Z], so serialize a [Z, Y, X] view in C order.
    vol = np.asarray(volume, dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {vol.shape}")
    arr = np.ascontiguousarray(vol.transpose(2, 1, 0))
    nx, ny, nz = vol.shape
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    struct.pack_into("<h", header, 70, 16)  # float32
    struct.pack_into("<h", header, 72, 32)
    struct.pack_into("<8f", header, 76, 1.0, spacing[0], spacing[1], spacing[2], 1.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<f", header, 112, 1.0)
    struct.pack_into("<f", header, 116, 0.0)
    struct.pack_into("<h", header, 252, 1)  # qform_code
    struct.pack_into("<h", header, 254, 1)  # sform_code
    struct.pack_into("<4f", header, 280, spacing[0], 0.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 296, 0.0, spacing[1], 0.0, 0.0)
    struct.pack_into("<4f", header, 312, 0.0, 0.0, spacing[2], 0.0)
    header[344:348] = b"n+1\0"
    with gzip.open(path, "wb") as handle:
        handle.write(header)
        handle.write(b"\0\0\0\0")
        handle.write(arr.tobytes(order="C"))


def norm_image(slice2d: np.ndarray, lo: float, hi: float) -> Image.Image:
    if hi <= lo:
        hi = lo + 1e-6
    data = np.clip((np.asarray(slice2d, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0)
    data = (255.0 * data).astype(np.uint8)
    return Image.fromarray(data, mode="L").convert("RGB")


def save_centers(volumes: dict[str, np.ndarray], out: Path, lo: float, hi: float) -> None:
    axes = {
        "x": lambda v: v[v.shape[0] // 2, :, :],
        "y": lambda v: v[:, v.shape[1] // 2, :],
        "z": lambda v: v[:, :, v.shape[2] // 2],
    }
    for name, vol in volumes.items():
        for axis, getter in axes.items():
            image = norm_image(np.rot90(getter(vol)), lo, hi)
            image.save(out / f"{name}_center_{axis}.png")

    # Side-by-side images make the three initializations directly comparable.
    for axis, getter in axes.items():
        tiles = [norm_image(np.rot90(getter(vol)), lo, hi) for vol in volumes.values()]
        canvas = Image.new("RGB", (tiles[0].width * len(tiles), tiles[0].height + 32), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (name, tile) in enumerate(zip(volumes, tiles)):
            canvas.paste(tile, (i * tile.width, 32))
            draw.text((i * tile.width + 8, 8), name, fill="black")
        canvas.save(out / f"comparison_center_{axis}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--min_component", type=int, default=64)
    parser.add_argument("--spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(np.load(args.input), dtype=np.float32)
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3D npy volume, got {raw.shape}")
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.maximum(raw, 0.0)

    # Scheme 1: light denoising only, with the same coordinate grid.
    denoised = gaussian_filter(raw, sigma=float(args.sigma), mode="nearest")
    denoised = np.maximum(denoised, 0.0).astype(np.float32)

    # Scheme 4: conservative screening. Normalize by p99.5, retain voxels
    # above the same 0.10 threshold used by the current initializer, then
    # remove isolated components smaller than min_component voxels.
    positive = raw[raw > 0]
    scale = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    normalized = raw / max(scale, 1e-12)
    mask = normalized >= float(args.threshold)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, count = label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= int(args.min_component)
    keep[0] = False
    mask = keep[labels]
    screened = np.where(mask, raw, 0.0).astype(np.float32)

    volumes = {"original": raw, "denoised": denoised, "screened": screened}
    lo = float(np.percentile(raw, 1.0))
    hi = float(np.percentile(raw, 99.5))
    save_centers(volumes, args.output, lo, hi)
    write_nifti(args.output / "vol_fbp_original.nii.gz", raw, tuple(args.spacing))
    write_nifti(args.output / "vol_fbp_denoised.nii.gz", denoised, tuple(args.spacing))
    write_nifti(args.output / "vol_fbp_screened.nii.gz", screened, tuple(args.spacing))
    np.save(args.output / "vol_fbp_denoised.npy", denoised)
    np.save(args.output / "vol_fbp_screened.npy", screened)
    np.save(args.output / "screen_mask.npy", mask.astype(np.uint8))
    (args.output / "stats.txt").write_text(
        "\n".join([
            f"input={args.input}", f"shape={raw.shape}", f"dtype={raw.dtype}",
            f"sigma={args.sigma}", f"threshold={args.threshold}",
            f"min_component={args.min_component}", f"p99.5_scale={scale}",
            f"components={count}", f"kept_voxels={int(mask.sum())}",
            f"kept_fraction={float(mask.mean())}", f"spacing={tuple(args.spacing)}",
        ]) + "\n", encoding="utf-8",
    )
    print(f"input shape={raw.shape}, p99.5={scale:.6g}")
    print(f"screened voxels={int(mask.sum())}/{mask.size} ({mask.mean()*100:.2f}%), components={count}")
    print(f"outputs: {args.output}")


if __name__ == "__main__":
    main()
