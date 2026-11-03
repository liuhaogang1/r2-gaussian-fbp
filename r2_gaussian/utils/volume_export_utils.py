"""CPU-only previews and streaming NIfTI output for XYZ Gaussian volumes."""

import gzip
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

NIFTI_XYZ_MARKER = "r2g axes=XYZ; LPS; dataset units"


def volume_geometry(shape, scanner):
    """Return XYZ spacing and first-voxel-center origin in dataset units."""
    shape = np.asarray(shape, dtype=np.float64)
    size = np.asarray(scanner.get("sVoxel"), dtype=np.float64)
    center = np.asarray(scanner.get("offOrigin", [0, 0, 0]), dtype=np.float64)
    if shape.shape != (3,) or not np.all(np.isfinite(shape)) or np.any(shape <= 0):
        raise ValueError("volume shape must contain three positive finite counts")
    if size.shape != (3,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError("scanner.sVoxel must contain three positive finite XYZ lengths")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("scanner.offOrigin must contain three finite XYZ coordinates")
    spacing = size / shape
    return spacing, center - size / 2 + spacing / 2


def save_nifti_xyz(volume, scanner, output_path):
    """Atomically replace an XYZ export using scanner voxel-center geometry."""
    spacing, origin = volume_geometry(volume.shape, scanner)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=output.parent, suffix=".nii.gz")
    os.close(descriptor)
    Path(temporary).unlink()
    try:
        save_nifti_streaming(volume, spacing, origin, temporary)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"axes": "XYZ", "spacing": spacing.tolist(), "origin_lps": origin.tolist(),
            "length_units": "dataset units (no conversion)", "format_version": 2}


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
    """Write float32 NIfTI with I/J/K equal to input X/Y/Z.

    Equivalent to SimpleITK.GetImageFromArray(volume.transpose(2, 1, 0)),
    followed by SetSpacing(spacing) and SetOrigin(origin). The description
    marks the axis convention so references can identify it during mask import.

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
    nifti_shape = tuple(volume.shape)

    # NIfTI-1 single-file header plus its four-byte extension flag. Using a
    # streaming gzip writer avoids a full transposed array and a SimpleITK copy.
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, *nifti_shape, 1, 1, 1, 1)
    struct.pack_into("<2h", header, 70, 16, 32)  # DT_FLOAT32, bitpix
    struct.pack_into("<8f", header, 76, 1.0, *spacing, 1.0, 1.0, 1.0, 1.0)
    struct.pack_into("<3f", header, 108, 352.0, 1.0, 0.0)  # offset, slope, intercept
    description = NIFTI_XYZ_MARKER.encode("ascii")
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
                # File I (input X) is fastest; each stored plane is [Y, X].
                plane = np.asarray(volume[:, :, index].T, dtype="<f4", order="C")
                stream.write(plane.tobytes(order="C"))
    except BaseException:
        raw.close()
        Path(output_path).unlink()
        raise
