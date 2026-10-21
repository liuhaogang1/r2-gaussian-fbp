"""CPU-only previews and streaming NIfTI output for XYZ Gaussian volumes."""

import gzip
import struct
from pathlib import Path

import numpy as np


def save_center_slices(volume, spacing, output_path):
    """Save a shared-window XYZ center-plane montage without copying the volume."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    center = tuple(size // 2 for size in volume.shape)
    planes = (volume[center[0], :, :], volume[:, center[1], :], volume[:, :, center[2]])
    finite_values = [plane[np.isfinite(plane)].astype(np.float32) for plane in planes]
    samples = np.concatenate(finite_values)
    if samples.size:
        low, high = np.percentile(samples, (1, 99))
        if high <= low:
            low, high = float(samples.min()), float(samples.max())
        if high <= low:
            high = low + 1.0
    else:
        low, high = 0.0, 1.0
    del samples, finite_values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    try:
        for axis, plane, normal, horizontal, vertical in zip(
            axes, planes, range(3), (1, 0, 0), (2, 2, 1)
        ):
            # imshow consumes [vertical, horizontal], while our planes retain XYZ order.
            image = axis.imshow(
                plane.T, origin="lower", cmap="gray", vmin=low, vmax=high,
                extent=(0, volume.shape[horizontal] * spacing[horizontal],
                        0, volume.shape[vertical] * spacing[vertical]),
                interpolation="nearest", aspect="equal",
            )
            axis.set_title(f"{'XYZ'[normal]} center (index {center[normal]})")
            axis.set_xlabel(f"{'XYZ'[horizontal]} (dataset length units, local)")
            axis.set_ylabel(f"{'XYZ'[vertical]} (dataset length units, local)")
        fig.colorbar(image, ax=axes, shrink=0.75, label="Gaussian density")
        fig.suptitle("Central slices | shared 1-99% display window")
        # Exclusive creation preserves the exporter's no-overwrite policy.
        with open(output_path, "xb") as handle:
            fig.savefig(handle, format="png", dpi=160)
    finally:
        plt.close(fig)
    return {"center_indices_xyz": list(center), "display_window": [float(low), float(high)]}


def save_nifti_streaming(volume, spacing, origin, output_path):
    """Write NIfTI-1 float32 in Z planes (XYZ input, X-fastest file order).

    spacing/origin use the original dataset length units, with identity LPS
    direction, like SimpleITK. NIfTI sform stores the equivalent RAS transform.
    No metric unit is asserted because dataset metadata does not specify one.
    Only one plane is converted to float32 at a time, including for float16 input.
    """
    if volume.ndim != 3 or any(size < 1 or size > 32767 for size in volume.shape):
        raise ValueError("NIfTI-1 requires three dimensions with sizes in [1, 32767]")
    spacing = np.asarray(spacing, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing must contain three positive finite values")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must contain three finite values")

    # NIfTI-1 single-file header plus its four-byte extension flag. Using a
    # streaming gzip writer avoids a full transposed array and a SimpleITK copy.
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *volume.shape, 1, 1, 1, 1)
    struct.pack_into("<2h", header, 70, 16, 32)  # DT_FLOAT32, bitpix
    struct.pack_into("<8f", header, 76, 1.0, *spacing, 1.0, 1.0, 1.0, 1.0)
    struct.pack_into("<3f", header, 108, 352.0, 1.0, 0.0)  # offset, slope, intercept
    description = b"Gaussian field; original dataset geometry; length units unspecified"
    header[148:148 + len(description)] = description
    struct.pack_into("<2h", header, 252, 0, 2)  # qform unknown, sform aligned
    struct.pack_into("<4f", header, 280, -spacing[0], 0.0, 0.0, -origin[0])
    struct.pack_into("<4f", header, 296, 0.0, -spacing[1], 0.0, -origin[1])
    struct.pack_into("<4f", header, 312, 0.0, 0.0, spacing[2], origin[2])
    header[344:348] = b"n+1\x00"

    # Opening outside the try ensures an existing file is never deleted.
    raw = open(output_path, "xb")
    try:
        with raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as stream:
            stream.write(header)
            for index in range(volume.shape[2]):
                plane = np.asarray(volume[:, :, index].T, dtype="<f4", order="C")
                stream.write(plane.tobytes(order="C"))
    except BaseException:
        raw.close()
        Path(output_path).unlink()
        raise
