"""Export a trained Gaussian field on a high-resolution grid, in bounded chunks.

This is an optional post-processing entry point. It does not change training,
the normal test workflow, or the pipeline launcher.
"""

import json
import os
import os.path as osp
from argparse import ArgumentParser
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

from r2_gaussian.arguments import ModelParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, initialize_gaussian, query
from r2_gaussian.utils.general_utils import safe_state, t2a
from r2_gaussian.utils.volume_export_utils import save_center_slices, save_nifti_streaming


_SCANNER_SPATIAL_KEYS = (
    "dVoxel",
    "sVoxel",
    "sDetector",
    "dDetector",
    "offOrigin",
    "offDetector",
    "DSD",
    "DSO",
)


def load_scanner_config(source_path):
    """Load scanner bounds with the same normalization used by Scene."""
    metadata_path = osp.join(source_path, "meta_data.json")
    if not osp.isfile(metadata_path):
        raise FileNotFoundError(
            f"Expected a Blender-style dataset metadata file: {metadata_path}"
        )
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    scanner = metadata.get("scanner")
    if not isinstance(scanner, dict):
        raise ValueError(f"Missing scanner configuration in {metadata_path}")

    # dataset_readers.readBlenderInfo normalizes the scene to [-1, 1] using
    # this scale before loading the trained Gaussian parameters.
    s_voxel = np.asarray(scanner.get("sVoxel"), dtype=np.float64)
    if s_voxel.shape != (3,) or not np.all(np.isfinite(s_voxel)) or np.any(s_voxel <= 0):
        raise ValueError("scanner.sVoxel must contain three positive finite values")
    scene_scale = 2.0 / float(np.max(s_voxel))
    if "dVoxel" not in scanner:
        scanner["dVoxel"] = (s_voxel / np.asarray(scanner["nVoxel"])).tolist()
    if "dDetector" not in scanner:
        scanner["dDetector"] = (
            np.asarray(scanner["sDetector"]) / np.asarray(scanner["nDetector"])
        ).tolist()
    for key in _SCANNER_SPATIAL_KEYS:
        if key in scanner:
            scanner[key] = (np.asarray(scanner[key], dtype=np.float64) * scene_scale).tolist()
    return scanner, scene_scale


def validate_triplet(name, values, positive=True):
    if len(values) != 3:
        raise ValueError(f"{name} must have exactly three values")
    result = tuple(int(value) for value in values)
    if positive and any(value <= 0 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


def export_volume(gaussians, pipeline, center, full_size, resolution, chunk_size, output_path, dtype):
    """Query and write a globally aligned volume without holding it on GPU."""
    resolution = validate_triplet("resolution", resolution)
    chunk_size = validate_triplet("chunk_size", chunk_size)
    full_size = np.asarray(full_size, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    spacing = full_size / np.asarray(resolution, dtype=np.float64)
    output_path = osp.abspath(output_path)
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    if osp.exists(output_path):
        raise FileExistsError(
            f"Output already exists: {output_path}. Choose another path or remove it explicitly."
        )

    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=resolution,
    )
    starts = [range(0, resolution[axis], chunk_size[axis]) for axis in range(3)]
    block_count = int(np.prod([len(axis_starts) for axis_starts in starts]))
    try:
        for start in tqdm(product(*starts), total=block_count, desc="Query Gaussian field", unit="block"):
            stop = tuple(min(start[axis] + chunk_size[axis], resolution[axis]) for axis in range(3))
            local_n = tuple(stop[axis] - start[axis] for axis in range(3))
            local_size = spacing * np.asarray(local_n, dtype=np.float64)
            # The block center is computed from the global lower bound and
            # global voxel spacing so adjacent calls share one exact grid.
            local_center = center - full_size / 2 + (
                np.asarray(start, dtype=np.float64)
                + np.asarray(local_n, dtype=np.float64) / 2
            ) * spacing
            with torch.inference_mode():
                result = query(
                    gaussians,
                    local_center.tolist(),
                    local_n,
                    local_size.tolist(),
                    pipeline,
                )["vol"]
                block = t2a(result).astype(dtype, copy=False)
            out[
                start[0] : stop[0],
                start[1] : stop[1],
                start[2] : stop[2],
            ] = block
            del result, block
        out.flush()
    except Exception:
        out.flush()
        del out
        # This file was created by this invocation; remove the incomplete
        # result so a later retry does not mistake it for a finished export.
        os.remove(output_path)
        raise
    del out
    return spacing


def main():
    parser = ArgumentParser(description="Optional chunked high-resolution Gaussian volume export")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1, help="Checkpoint iteration; -1 selects the latest")
    parser.add_argument("--resolution", nargs=3, type=int, default=(1024, 1024, 1024), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--chunk_size", nargs=3, type=int, default=(128, 128, 64), metavar=("CX", "CY", "CZ"))
    parser.add_argument("--output", default=None, help="Output .npy; defaults under <model_path>/highres/")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--pose_enabled", action="store_true", help="Use the pose-calibration voxelizer backend")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    resolution = validate_triplet("resolution", args.resolution)
    chunk_size = validate_triplet("chunk_size", args.chunk_size)
    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)
    pipeline_params.pose_enabled = args.pose_enabled

    safe_state(args.quiet)
    scanner_cfg, scene_scale = load_scanner_config(dataset.source_path)
    center = scanner_cfg.get("offOrigin")
    full_size = scanner_cfg.get("sVoxel")
    if center is None or full_size is None:
        raise ValueError("scanner configuration must define offOrigin and sVoxel")

    gaussians = GaussianModel(None)
    loaded_iter = initialize_gaussian(gaussians, dataset, args.iteration)
    if loaded_iter is None:
        raise RuntimeError("Could not determine the loaded checkpoint iteration")

    output_path = args.output or osp.join(
        dataset.model_path,
        "highres",
        f"vol_pred_iter_{loaded_iter}_{resolution[0]}x{resolution[1]}x{resolution[2]}.npy",
    )
    output_path = osp.abspath(output_path)
    output_stem = osp.splitext(output_path)[0]
    metadata_path = output_stem + ".json"
    slices_path = output_stem + "_center_slices.png"
    nifti_path = output_stem + ".nii.gz"
    # Fail before an expensive query if any of the automatically generated
    # companion files would overwrite a previous result.
    for path in (output_path, metadata_path, slices_path, nifti_path):
        if osp.exists(path):
            raise FileExistsError(f"Output already exists: {path}. Choose another --output path.")
    spacing = export_volume(
        gaussians,
        pipeline_params,
        center,
        full_size,
        resolution,
        chunk_size,
        output_path,
        args.dtype,
    )
    # Query coordinates are normalized model coordinates. Restore the original
    # dataset geometry for Slicer; density values are deliberately unchanged.
    dataset_spacing = spacing / scene_scale
    dataset_center = np.asarray(center, dtype=np.float64) / scene_scale
    dataset_size = np.asarray(full_size, dtype=np.float64) / scene_scale
    dataset_origin = dataset_center - dataset_size / 2 + dataset_spacing / 2
    # Match test.py's NIfTI-only XY permutation. Query, NPY and center-slice
    # PNG retain their original XYZ convention; spacing/origin are unchanged.
    nifti_shape = np.asarray(resolution, dtype=np.int64)[[1, 0, 2]]
    nifti_size = nifti_shape * dataset_spacing
    nifti_center = dataset_origin + (nifti_shape - 1) * dataset_spacing / 2
    volume = np.load(output_path, mmap_mode="r")
    try:
        print("Saving X/Y/Z center slices...")
        preview_info = save_center_slices(volume, dataset_spacing, slices_path)
        print("Writing compressed NIfTI one Z plane at a time...")
        save_nifti_streaming(volume, dataset_spacing, dataset_origin, nifti_path)
    finally:
        del volume
    metadata = {
        "volume": osp.basename(output_path),
        "center_slices": osp.basename(slices_path),
        "nifti": osp.basename(nifti_path),
        "preview": preview_info,
        "shape_xyz": list(resolution),
        "dtype": args.dtype,
        "spacing_xyz": spacing.tolist(),
        "center_xyz": [float(value) for value in center],
        "size_xyz": [float(value) for value in full_size],
        "scene_scale": float(scene_scale),
        "nifti_geometry": {
            "shape_xyz": nifti_shape.tolist(),
            "input_axes_for_nifti_ijk": ["Y", "X", "Z"],
            "axis_convention": "legacy test.py: GetImageFromArray(volume.transpose(2, 0, 1))",
            "spacing_xyz": dataset_spacing.tolist(),
            "origin_xyz": dataset_origin.tolist(),
            "center_xyz": nifti_center.tolist(),
            "size_xyz": nifti_size.tolist(),
            "coordinate_system": "LPS (identity direction); stored as RAS sform",
            "length_unit": "original dataset units (unspecified in NIfTI header)",
            "dtype": "float32",
            "intensity": "unchanged Gaussian density, matching the NPY",
        },
        "model_path": osp.abspath(dataset.model_path),
        "source_path": osp.abspath(dataset.source_path),
        "iteration": int(loaded_iter),
        "chunk_size_xyz": list(chunk_size),
        "pose_enabled": bool(args.pose_enabled),
        "note": "Direct samples of the trained Gaussian field; not a new reconstruction or super-resolution inference.",
    }
    with open(metadata_path, "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Saved high-resolution volume: {output_path}")
    print(f"Saved center slices: {slices_path}")
    print(f"Saved Slicer volume: {nifti_path}")
    print(f"Saved volume metadata: {metadata_path}")


if __name__ == "__main__":
    main()
