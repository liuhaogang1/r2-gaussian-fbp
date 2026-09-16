"""Apply a 3D Slicer mask to an FBP volume.

Mask voxels with value > 0 keep the original FBP density.  Mask voxels with
value == 0 are set to zero.  The output remains a 3D NumPy volume and can be
passed directly to ``init_from_fbp.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_slicer_mask(mask_path: Path, reference_path: Path | None):
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as error:
        raise SystemExit(
            "SimpleITK is required. Activate the environment used for FBP "
            "preprocessing and run this script again."
        ) from error

    mask_image = sitk.ReadImage(str(mask_path))
    if reference_path is not None:
        reference_image = sitk.ReadImage(str(reference_path))
        mask_image = sitk.Resample(
            mask_image,
            reference_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )

    mask_zyx = np.asarray(sitk.GetArrayFromImage(mask_image))
    if mask_zyx.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got {mask_zyx.shape}: {mask_path}")

    # prepare_fbp_tiff.py writes the [X, Y, Z] volume as a SimpleITK array in
    # [Z, X, Y] order. Convert the labelmap back to the NumPy volume order.
    return mask_zyx.transpose(1, 2, 0) > 0, mask_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--volume",
        type=Path,
        required=True,
        help="Input 3D FBP volume, for example vol_fbp.npy.",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        required=True,
        help="3D Slicer labelmap, for example sample_mask.nii.gz.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Reference NIfTI used in Slicer, normally vol_fbp_preview.nii.gz. "
            "When supplied, the mask is nearest-neighbor resampled to it."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output masked 3D volume, for example original_masked.npy.",
    )
    parser.add_argument(
        "--nii_output",
        type=Path,
        default=None,
        help=(
            "Optional Slicer-viewable NIfTI output. Defaults to the same folder "
            "as --output with the name original_masked.nii.gz."
        ),
    )
    args = parser.parse_args()

    volume = np.asarray(np.load(args.volume))
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got {volume.shape}: {args.volume}")

    mask, reference_image = load_slicer_mask(args.mask, args.reference)
    if mask.shape != volume.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match volume shape {volume.shape}. "
            "Check that Slicer used vol_fbp_preview.nii.gz as its reference."
        )
    if not np.any(mask):
        raise ValueError("The mask contains no nonzero sample voxels")

    masked_volume = np.where(mask, volume, np.zeros((), dtype=volume.dtype))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, masked_volume)

    nii_output = args.nii_output
    if nii_output is None:
        nii_output = args.output.with_name("original_masked.nii.gz")

    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as error:
        raise SystemExit(
            "SimpleITK is required to write the Slicer-viewable NIfTI output."
        ) from error

    # The NumPy volume uses [X, Y, Z], while SimpleITK uses [Z, X, Y].
    masked_zyx = np.asarray(masked_volume).transpose(2, 0, 1)
    masked_image = sitk.GetImageFromArray(masked_zyx)
    masked_image.CopyInformation(reference_image)
    nii_output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(masked_image, str(nii_output))

    print(f"Input volume: {args.volume}")
    print(f"Input shape: {volume.shape}; dtype: {volume.dtype}")
    print(f"Mask voxels kept: {int(mask.sum())} ({mask.mean():.4%})")
    print(f"Nonzero voxels before: {int(np.count_nonzero(volume))}")
    print(f"Nonzero voxels after: {int(np.count_nonzero(masked_volume))}")
    print(f"Saved: {args.output}")
    print(f"Saved Slicer reference volume: {nii_output}")


if __name__ == "__main__":
    main()
