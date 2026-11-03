"""Apply a 3D Slicer mask to an FBP volume.

Mask voxels with value > 0 keep the original FBP density.  Mask voxels with
value == 0 are set to zero.  The output remains a 3D NumPy volume and can be
passed directly to ``init_from_fbp.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from r2_gaussian.utils.volume_export_utils import NIFTI_XYZ_MARKER, save_nifti_xyz


def resolve_mask_axes(image, axis_order, volume_shape):
    if axis_order not in ("auto", "xyz", "legacy_yxz"):
        raise ValueError(f"Unknown mask axis convention: {axis_order}")
    if axis_order != "auto":
        return axis_order
    if image.HasMetaDataKey("descrip") and "r2g axes=XYZ" in image.GetMetaData("descrip"):
        return "xyz"
    if volume_shape is not None:
        size = image.GetSize()
        shape = tuple(volume_shape)
        candidates = [name for name, expected in (
            ("xyz", shape), ("legacy_yxz", (shape[1], shape[0], shape[2]))) if size == expected]
        if len(candidates) == 1:
            return candidates[0]
    raise ValueError(
        "Cannot determine mask axes from an unmarked or ambiguous reference. "
        "Use --mask_axes xyz for XYZ exports, or --mask_axes legacy_yxz for old FBP/test exports. "
        "For cropped or reoriented labelmaps, also supply the original --reference NIfTI."
    )


def load_slicer_mask(mask_path: Path, reference_path: Path | None,
                     axis_order="auto", volume_shape=None):
    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as error:
        raise SystemExit(
            "SimpleITK is required. Activate the environment used for FBP "
            "preprocessing and run this script again."
        ) from error

    mask_image = sitk.ReadImage(str(mask_path))
    if mask_image.GetDimension() != 3 or mask_image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"Expected a scalar 3D labelmap: {mask_path}")
    reference_image = sitk.ReadImage(str(reference_path)) if reference_path is not None else mask_image
    if reference_image.GetDimension() != 3:
        raise ValueError("Reference must be a 3D image")
    # The reference defines the voxel-index convention; the labelmap may have
    # been cropped/reoriented by Slicer and is resampled in physical space.
    axes = resolve_mask_axes(reference_image, axis_order, volume_shape)
    expected = None if volume_shape is None else tuple(volume_shape)
    if expected is not None and axes == "legacy_yxz":
        expected = (expected[1], expected[0], expected[2])
    if expected is not None and reference_image.GetSize() != expected:
        raise ValueError(f"Reference size {reference_image.GetSize()} does not match {expected}. "
                         "Use the original full-volume --reference in the selected --mask_axes convention.")
    # Binarize before conversion to uint8 so label IDs >=256 are retained.
    mask_image = sitk.Cast(mask_image > 0, sitk.sitkUInt8)
    if reference_path is not None:
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

    reference_image.SetMetaData("r2g_input_axes", axes)
    permutation = (2, 1, 0) if axes == "xyz" else (1, 2, 0)
    return mask_zyx.transpose(permutation) > 0, reference_image


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
    parser.add_argument("--mask_axes", choices=("auto", "xyz", "legacy_yxz"), default="auto",
                        help="Reference voxel axes. Unmarked cubic old files require an explicit choice.")
    parser.add_argument("--metadata", type=Path,
                        help="Scanner meta_data.json; defaults beside --volume. Restores dataset geometry.")
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

    mask, reference_image = load_slicer_mask(args.mask, args.reference, args.mask_axes, volume.shape)
    if mask.shape != volume.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match volume shape {volume.shape}. "
            "Check --reference and --mask_axes."
        )
    if not np.any(mask):
        raise ValueError("The mask contains no nonzero sample voxels")

    from r2_gaussian.utils.artifact_utils import npy_path, atomic_npy
    args.output = npy_path(args.output)
    nii_output = args.nii_output
    if nii_output is None:
        nii_output = args.output.with_name("original_masked.nii.gz")

    try:
        import SimpleITK as sitk
    except ModuleNotFoundError as error:
        raise SystemExit(
            "SimpleITK is required to write the Slicer-viewable NIfTI output."
        ) from error

    metadata_path = args.metadata or args.volume.parent / "meta_data.json"
    if args.metadata is not None and not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    scanner = None
    if metadata_path.exists():
        scanner = json.loads(metadata_path.read_text(encoding="utf-8"))["scanner"]
        if tuple(scanner.get("nVoxel", volume.shape)) != volume.shape:
            raise ValueError("Volume shape differs from scanner.nVoxel")
    protected = {p.resolve() for p in (args.volume, args.mask, args.reference, metadata_path) if p is not None}
    if args.output.resolve() in protected or nii_output.resolve() in protected or args.output.resolve() == nii_output.resolve():
        raise ValueError("Mask outputs must not overwrite source files or each other")
    masked_volume = np.where(mask, volume, np.zeros((), dtype=volume.dtype))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nii_output.parent.mkdir(parents=True, exist_ok=True)
    if scanner is not None:
        save_nifti_xyz(masked_volume, scanner, nii_output)
    else:
        # Without scanner lengths, preserve the reference's physical mapping
        # while reindexing the payload into XYZ. Never invent dataset geometry.
        masked_image = sitk.GetImageFromArray(masked_volume.transpose(2, 1, 0))
        order = [0, 1, 2] if reference_image.GetMetaData("r2g_input_axes") == "xyz" else [1, 0, 2]
        masked_image.SetSpacing(tuple(np.asarray(reference_image.GetSpacing())[order]))
        masked_image.SetOrigin(reference_image.GetOrigin())
        direction = np.asarray(reference_image.GetDirection()).reshape(3, 3)[:, order]
        masked_image.SetDirection(tuple(direction.ravel()))
        masked_image.SetMetaData("descrip", NIFTI_XYZ_MARKER)
        sitk.WriteImage(masked_image, str(nii_output))

    atomic_npy(args.output, masked_volume)
    print(f"Input volume: {args.volume}")
    print(f"Input shape: {volume.shape}; dtype: {volume.dtype}")
    print(f"Mask voxels kept: {int(mask.sum())} ({mask.mean():.4%})")
    print(f"Nonzero voxels before: {int(np.count_nonzero(volume))}")
    print(f"Nonzero voxels after: {int(np.count_nonzero(masked_volume))}")
    print(f"Saved: {args.output}")
    print(f"Saved Slicer reference volume: {nii_output}")


if __name__ == "__main__":
    main()
