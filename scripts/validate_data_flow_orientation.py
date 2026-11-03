"""Audit array axes, initialization coordinates, previews and NIfTI geometry.

Run in the project's CUDA/TIGRE environment. Writes only to a new audit folder;
production inputs, models and export conventions are never modified.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def image_info(path):
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    return image, {
        "path": str(path), "size_ijk": list(image.GetSize()),
        "spacing_ijk": list(image.GetSpacing()),
        "origin_lps": list(image.GetOrigin()),
        "direction_lps": list(image.GetDirection()),
    }


def capture_images(function):
    from matplotlib.figure import Figure

    recorded = []
    savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        recorded.extend((np.asarray(artist.get_array()).copy(), artist.origin)
                        for axis in figure.axes for artist in axis.images)
        return savefig(figure, *args, **kwargs)

    with mock.patch.object(Figure, "savefig", new=capture):
        function()
    return recorded


def audit_exports(out, checks):
    import SimpleITK as sitk
    from PIL import Image
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import prepare_fbp_tiff as prepare
    import init_from_fbp as init
    from scripts import init_denoise as denoise
    from scripts import local_fbp_preview as local
    from r2_gaussian.utils.volume_export_utils import save_nifti_streaming

    shape = (9, 11, 13)
    x, y, z = np.indices(shape)
    volume = (1 + 100 * x + 10 * y + z).astype(np.float32)
    scanner = {"sVoxel": [1.8, 3.3, 5.2], "offOrigin": [.25, -.4, .9]}
    spacing = np.asarray(scanner["sVoxel"]) / shape
    origin = np.asarray(scanner["offOrigin"]) - np.asarray(scanner["sVoxel"]) / 2 + spacing / 2
    folder = out / "export_conventions"
    folder.mkdir()
    np.save(folder / "coded_xyz.npy", volume)
    low, high = prepare.save_fbp_preview(folder, volume, scanner["sVoxel"], off_origin=scanner["offOrigin"])
    init.save_fbp_nifti(volume, folder / "init_xyz.nii.gz", scanner)
    denoise.save_nifti_xyz(volume, scanner, folder / "denoise_xyz.nii.gz")
    save_nifti_streaming(volume, spacing, origin, folder / "highres_xyz.nii.gz")
    local.write_nifti(folder / "local_xyz.nii.gz", volume, tuple(spacing), origin)
    geometries = {}
    landmark = (2, 7, 4)
    expected_world = origin + np.asarray(landmark) * spacing
    for name, filename, ijk in (
        ("fbp", "vol_fbp_preview.nii.gz", landmark),
        ("init_xyz", "init_xyz.nii.gz", landmark),
        ("denoise_xyz", "denoise_xyz.nii.gz", landmark),
        ("highres_xyz", "highres_xyz.nii.gz", landmark),
        ("local_xyz", "local_xyz.nii.gz", landmark),
    ):
        image, info = image_info(folder / filename)
        assert image.GetPixel(ijk) == float(volume[landmark]), name
        assert image.GetSize() == shape
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image).transpose(2, 1, 0), volume)
        world = image.TransformIndexToPhysicalPoint(ijk)
        np.testing.assert_allclose(world, expected_world, atol=1e-6)
        np.testing.assert_allclose(image.GetDirection(), np.eye(3).ravel(), atol=1e-6)
        info.update(landmark_xyz=list(landmark), landmark_ijk=list(ijk), landmark_lps=list(world),
                    expected_scanner_lps=expected_world.tolist(),
                    landmark_world_error=float(np.linalg.norm(world - expected_world)))
        geometries[name] = info
    np.testing.assert_allclose(geometries["denoise_xyz"]["landmark_lps"], expected_world, atol=1e-6)
    checks["nifti_payload_roundtrips"] = "5/5 exact XYZ; same shape, LPS direction and physical landmark coordinates"
    checks["nifti_world_coordinates"] = "all exporters match scanner voxel centers"
    from apply_mask_to_volume import load_slicer_mask

    legacy_image = sitk.GetImageFromArray(volume.transpose(2, 0, 1))
    sitk.WriteImage(legacy_image, str(folder / "legacy_reference.nii.gz"))
    legacy_mask, _ = load_slicer_mask(folder / "legacy_reference.nii.gz", None,
                                      "legacy_yxz", shape)
    xyz_mask, _ = load_slicer_mask(folder / "denoise_xyz.nii.gz", None, volume_shape=shape)
    assert legacy_mask.shape == shape
    np.testing.assert_array_equal(xyz_mask, volume > 0)
    cube = np.zeros((9, 9, 9), dtype=np.float32)
    cube[1, 6, 4], cube[7, 2, 5] = 1., 2.
    denoise.save_nifti_xyz(cube, scanner, folder / "cubic_xyz_mask.nii.gz")
    cubic_mask, _ = load_slicer_mask(folder / "cubic_xyz_mask.nii.gz", None, volume_shape=cube.shape)
    np.testing.assert_array_equal(cubic_mask, cube > 0)
    sitk.WriteImage(sitk.GetImageFromArray(cube.transpose(2, 0, 1)), str(folder / "cubic_legacy.nii.gz"))
    legacy_cube, _ = load_slicer_mask(folder / "cubic_legacy.nii.gz", None, "legacy_yxz", cube.shape)
    np.testing.assert_array_equal(legacy_cube, cube > 0)
    try:
        load_slicer_mask(folder / "cubic_legacy.nii.gz", None, volume_shape=cube.shape)
    except ValueError as error:
        assert "ambiguous" in str(error)
    else:
        raise AssertionError("Unmarked cubic legacy data must require explicit axes")
    checks["mask_import_convention"] = {
        "legacy_export_mask_shape": list(legacy_mask.shape),
        "xyz_export_mask_shape": list(xyz_mask.shape),
        "expected_volume_shape": list(shape),
        "cubic_mask": "XYZ and explicit legacy both preserve landmarks; ambiguous auto import rejected",
    }

    planes = [np.take(volume, n // 2, axis=a) for a, n in enumerate(shape)]
    display = np.clip((volume - low) / (high - low), 0, 1)
    for axis, name in enumerate("xyz"):
        np.testing.assert_array_equal(np.asarray(Image.open(folder / f"vol_fbp_preview_{name}.png")),
                                      np.round(np.flipud(np.take(display, shape[axis] // 2, axis=axis).T) * 255).astype(np.uint8))
    selected = np.argwhere(volume > 0)
    init_images = capture_images(lambda: init.save_initialization_slices(
        shape, selected, volume[tuple(selected.T)], folder / "init_slices.png", positions=(.5,)))
    denoise_folder = folder / "denoise_previews"
    denoise_folder.mkdir()
    denoise_images = capture_images(lambda: denoise.save_previews(
        volume, volume, volume > 0, denoise_folder))
    for axis in range(3):
        np.testing.assert_array_equal(init_images[axis][0], planes[axis].T)
        assert init_images[axis][1] == "lower"
        np.testing.assert_array_equal(denoise_images[axis * 5][0], planes[axis].T)
        assert denoise_images[axis * 5][1] == "lower"
    local.save_centers({"coded": volume}, folder, 0, float(volume.max()))
    for axis, name in enumerate("xyz"):
        np.testing.assert_array_equal(np.asarray(Image.open(folder / f"coded_center_{name}.png")),
                                      np.asarray(local.norm_image(np.rot90(planes[axis]), 0, float(volume.max()))))
        np.testing.assert_array_equal(np.rot90(planes[axis]), np.flipud(planes[axis].T))
    checks["preview_artist_and_png_arrays"] = "all three axes verified; all current previews share transpose/lower raster orientation"
    plane = np.zeros_like(planes[2])
    plane[1, 1:5] = .5
    plane[1:4, 1] = .5
    plane[6:8, 7:10] = 1.
    plane[4, 7] = .75
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    conventions = [(plane.T, "lower", "FBP preview: fixed", "X", "Y"),
                   (plane.T, "lower", "Init slices: fixed", "X", "Y"),
                   (plane.T, "lower", "Denoise: transpose, lower", "X", "Y"),
                   (np.rot90(plane), "upper", "Local preview: rot90", "X", "PNG row (Y decreases)")]
    for column, (axis, (array, view_origin, title, horizontal, vertical)) in enumerate(zip(axes, conventions)):
        axis.imshow(array, origin=view_origin, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for x, y, label in ((1, 2, "A"), (6, 8, "B"), (4, 7, "C")):
            if column == 3:
                u, v = x, plane.shape[1] - 1 - y
            else:
                u, v = x, y
            axis.text(u, v, label, color="#ff7e67", ha="center", va="center", fontsize=13, weight="bold")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(horizontal)
        axis.set_ylabel(vertical)
    fig.suptitle("After repair: same XYZ marker orientation across all previews")
    fig.savefig(out / "preview_orientation_comparison.png", dpi=140)
    plt.close(fig)
    return geometries


def audit_pipeline(out, checks):
    import tifffile
    from scipy.ndimage import gaussian_filter
    from tigre.utilities.Ax import Ax
    import prepare_fbp_tiff as prepare
    from scripts import init_denoise as denoise

    raw_dir, dataset = out / "synthetic_tiffs", out / "dataset"
    raw_dir.mkdir()
    arguments = ["--input_dir", str(raw_dir), "--output_dir", str(dataset),
                 "--input_type", "line_integral", "--shift_v", "0", "--clip_percentile", "100",
                 "--pixel_subsample", "1", "--projection_scale", "1", "--pixel_size", "0.0625",
                 "--angle_interval", "3", "--n_train", "20", "--n_test", "8",
                 "--nVoxel", "33", "41", "49", "--sVoxel", "2", "2", "2"]
    args = prepare.build_parser().parse_args(arguments)
    args.offDetector = [.125, -.0625]
    shape = tuple(args.nVoxel)
    coords = np.moveaxis(np.indices(shape), 0, -1)
    world = (coords + .5) / shape * 2 - 1
    center = np.array([.24, -.17, .29])
    volume = np.exp(-.5 * np.sum(((world - center) / .1) ** 2, axis=-1)).astype(np.float32)
    angles = (np.arange(60) * np.pi / 60).astype(np.float32)
    geo = prepare.build_geometry(args, (48, 64))
    projections = Ax(np.ascontiguousarray(volume.transpose(2, 1, 0)), geo, angles)[:, ::-1, :]
    for index, projection in enumerate(projections):
        tifffile.imwrite(raw_dir / f"Tomo_{index:03d}.tiff", projection)
    with contextlib.redirect_stdout(io.StringIO()) as log:
        prepare.main(args)
    (out / "prepare.log").write_text(log.getvalue(), encoding="utf-8")
    meta = json.loads((dataset / "meta_data.json").read_text())
    for record in meta["proj_train"] + meta["proj_test"]:
        index = int(Path(record["file_path"]).stem)
        np.testing.assert_array_equal(np.load(dataset / record["file_path"]), projections[index])
    fbp = np.load(dataset / "vol_fbp.npy")
    from r2_gaussian.dataset.dataset_readers import readBlenderInfo

    with contextlib.redirect_stdout(io.StringIO()):
        loaded = readBlenderInfo(str(dataset), eval=True)
    np.testing.assert_array_equal(loaded.vol.detach().cpu().numpy(), fbp)
    for camera in loaded.train_cameras + loaded.test_cameras:
        np.testing.assert_array_equal(camera.image, np.load(camera.image_path) * loaded.scene_scale)
    checks["dataset_reader_arrays"] = "FBP loaded to CUDA unchanged; 28 camera images only intensity-scaled, with no spatial transpose/flip"
    bright = fbp > .5 * fbp.max()
    recovered = world[bright].mean(axis=0)
    np.testing.assert_allclose(recovered, center, atol=.07)
    checks["actual_tigre_roundtrip"] = {"input_center_xyz": center.tolist(), "reconstructed_center_xyz": recovered.tolist(),
                                          "grid_xyz": list(shape), "views": len(angles)}
    checks["tiff_to_saved_train_test_projections"] = "28 views bit-exact, no unintended row or column flip"
    source_hash = denoise.file_digest(dataset / "vol_fbp.npy")
    result = subprocess.run([sys.executable, str(ROOT / "init_from_fbp.py"),
        "--volume", str(dataset / "vol_fbp.npy"), "--denoise", "--denoise_save_nii",
        "--n_points", "400", "--density_thresh", "0.1", "--density_rescale", "0.25",
        "--seed", "7", "--visualize_slices"], cwd=ROOT, capture_output=True, text=True)
    (out / "init.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    folder = dataset / "init_denoise"
    filtered = np.load(folder / "vol_init_denoised.npy")
    expected = gaussian_filter(np.maximum(fbp, 0), .8, mode="nearest", truncate=4)
    np.testing.assert_array_equal(filtered, expected)
    assert denoise.file_digest(dataset / "vol_fbp.npy") == source_hash
    points = np.load(folder / "init_points.npy")
    scanner = meta["scanner"]
    size = np.array(scanner["sVoxel"])
    offset = np.array(scanner["offOrigin"])
    scene_scale = 2 / size.max()
    recovered_indices = ((points[:, :3] / scene_scale - offset) / size + .5) * shape - .5
    selected = np.rint(recovered_indices).astype(int)
    np.testing.assert_allclose(recovered_indices, selected, atol=3e-5)
    normalized = np.clip(filtered / np.percentile(filtered[filtered > 0], 99.5), 0, 1)
    np.testing.assert_allclose(points[:, 3], normalized[tuple(selected.T)] * .25, atol=1e-7)
    checks["fbp_to_denoised_npy"] = "bit-exact scipy reference; same XYZ grid; source SHA256 unchanged"
    checks["point_cloud_inverse_coordinate_mapping"] = "400/400 map back to integer XYZ voxel centers and matching densities"
    return scanner, folder / "init_points.npy"


def audit_gaussian(out, scanner, points_path, checks):
    import torch
    from r2_gaussian.gaussian import GaussianModel, initialize_gaussian, query
    from r2_gaussian.utils.geometry_utils import initialization_positions
    from export_highres import export_volume

    gaussians = GaussianModel()
    initialize_gaussian(gaussians, SimpleNamespace(ply_path=str(points_path)))
    np.testing.assert_array_equal(gaussians.get_xyz.detach().cpu().numpy(), np.load(points_path)[:, :3])
    checks["point_cloud_to_gaussian_model"] = "XYZ positions preserved exactly by production initializer"
    shape = (17, 21, 25)
    scanner = {"sVoxel": [1.7, 4.2, 7.5], "offOrigin": [.3, -.5, 1.1]}
    indices = np.array([[3, 5, 7], [13, 15, 18], [8, 12, 11]])
    positions = initialization_positions(indices, shape, scanner)
    scale = 2 / max(scanner["sVoxel"])
    size = np.array(scanner["sVoxel"]) * scale
    center = np.array(scanner["offOrigin"]) * scale
    sigma = size / shape * .55
    model = GaussianModel()
    model._xyz = torch.nn.Parameter(torch.tensor(positions, device="cuda"))
    model._scaling = torch.nn.Parameter(torch.tensor(np.tile(np.log(sigma), (3, 1)), dtype=torch.float32, device="cuda"))
    model._rotation = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]] * 3, device="cuda"))
    model._density = torch.nn.Parameter(model.density_inverse_activation(torch.tensor([[1.], [2.], [3.]], device="cuda")))
    pipe = SimpleNamespace(pose_enabled=False, compute_cov3D_python=False, debug=False)
    with torch.inference_mode():
        dense = query(model, center, shape, size, pipe)["vol"].cpu().numpy()
    assert dense.shape == shape
    for index in indices:
        lower = np.maximum(index - 2, 0)
        upper = np.minimum(index + 3, shape)
        block = dense[tuple(slice(a, b) for a, b in zip(lower, upper))]
        found = lower + np.array(np.unravel_index(block.argmax(), block.shape))
        np.testing.assert_array_equal(found, index)
    path = out / "chunked_gaussian_xyz.npy"
    export_volume(model, pipe, center, size, shape, (8, 9, 10), str(path), "float32")
    chunked = np.load(path)
    for index in indices:
        lower = np.maximum(index - 2, 0)
        upper = np.minimum(index + 3, shape)
        block = chunked[tuple(slice(a, b) for a, b in zip(lower, upper))]
        found = lower + np.array(np.unravel_index(block.argmax(), block.shape))
        np.testing.assert_array_equal(found, index)
        np.testing.assert_allclose(chunked[tuple(index)], dense[tuple(index)], rtol=5e-5)
    difference = np.abs(chunked - dense)
    np.testing.assert_allclose(chunked, dense, rtol=5e-5, atol=5e-6)
    checks["cuda_query_landmarks"] = "3/3 landmarks peak at original XYZ indices in both dense and chunked query; non-cubic offset grid"
    checks["dense_vs_chunked_query"] = {
        "max_abs_error": float(difference.max()),
        "relative_l1_error": float(difference.sum() / np.abs(dense).sum()),
        "count_over_5e_minus5": int(np.count_nonzero(difference > 5e-5)),
        "total_voxels": dense.size, "axis_order": "XYZ",
        "result": "Aligned CUDA tile grids remove chunk-boundary tail discrepancies; residuals meet rtol=5e-5, atol=5e-6.",
    }
    # Exercise the complete high-resolution CLI, including checkpoint loading,
    # sidecar metadata and the NIfTI path rather than only its shared writer.
    source, model_path = out / "query_dataset", out / "query_model"
    source.mkdir()
    metadata_scanner = dict(scanner, nVoxel=list(shape), dDetector=[1., 1.])
    (source / "meta_data.json").write_text(json.dumps({"scanner": metadata_scanner}))
    model.save_ply(str(model_path / "point_cloud" / "iteration_1" / "point_cloud.pickle"))
    cli_path = out / "highres_cli.npy"
    result = subprocess.run([sys.executable, str(ROOT / "export_highres.py"),
        "-s", str(source), "-m", str(model_path), "--iteration", "1",
        "--resolution", *map(str, shape), "--chunk_size", "8", "9", "10",
        "--output", str(cli_path)], cwd=ROOT, capture_output=True, text=True)
    (out / "highres_cli.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    result.check_returncode()
    cli_volume = np.load(cli_path)
    np.testing.assert_allclose(cli_volume, dense, rtol=5e-5, atol=5e-6)
    import SimpleITK as sitk
    exported = sitk.ReadImage(str(cli_path.with_suffix(".nii.gz")))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(exported).transpose(2, 1, 0), cli_volume)
    np.testing.assert_allclose(exported.GetSpacing(), np.asarray(scanner["sVoxel"]) / shape)
    expected_origin = np.asarray(scanner["offOrigin"]) - np.asarray(scanner["sVoxel"]) / 2 + np.asarray(scanner["sVoxel"]) / shape / 2
    np.testing.assert_allclose(exported.GetOrigin(), expected_origin, atol=1e-6)
    sidecar = json.loads(cli_path.with_suffix(".json").read_text())
    assert sidecar["nifti_geometry"]["input_axes_for_nifti_ijk"] == ["X", "Y", "Z"]
    assert sidecar["query_chunk_size_xyz"] == [8, 8, 8]
    checks["highres_cli"] = "Checkpoint loaded; CUDA query, XYZ NPY/NIfTI, original dataset spacing/origin and sidecar axis metadata agree"


def audit_existing(checks):
    import SimpleITK as sitk
    from PIL import Image

    folder = ROOT / "fbp_preview_local_20261028"
    path = folder / "vol_fbp_denoised.npy"
    if not path.exists():
        return
    volume = np.load(path, mmap_mode="r")
    image, info = image_info(folder / "vol_fbp_denoised.nii.gz")
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image).transpose(2, 1, 0), volume)
    original_image = sitk.ReadImage(str(folder / "vol_fbp_original.nii.gz"))
    raw = sitk.GetArrayFromImage(original_image).transpose(2, 1, 0)
    from scipy.ndimage import gaussian_filter
    np.testing.assert_array_equal(volume, gaussian_filter(raw, sigma=.8, mode="nearest"))
    low, high = np.percentile(raw, [1, 99.5])
    from scripts.local_fbp_preview import norm_image
    for axis, name in enumerate("xyz"):
        plane = np.take(volume, volume.shape[axis] // 2, axis=axis)
        np.testing.assert_array_equal(np.asarray(Image.open(folder / f"denoised_center_{name}.png")),
                                      np.asarray(norm_image(np.rot90(plane), low, high)))
    checks["existing_local_denoise_outputs"] = {"npy_vs_nifti": "bit-exact XYZ",
        "filter_vs_original": "bit-exact Gaussian filter, no transpose/flip",
        "pngs": "3/3 match rot90(center_plane) exactly", "nifti_geometry": info}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checks = {}
    report = {"python": sys.executable, "checks": checks, "limitations": [
        "Synthetic parallel-beam data; no experimental raw data or full training used.",
        "test.py CLI not run locally; actual evaluate_volume body tested separately without changing compatibility code.",
        "Optional pose CUDA backend not exercised."]}
    try:
        report["nifti_conventions"] = audit_exports(out, checks)
        scanner, points = audit_pipeline(out, checks)
        audit_gaussian(out, scanner, points, checks)
        audit_existing(checks)
        report["status"] = "orientation_checks_passed_unified_xyz_exports"
    except Exception:
        import traceback

        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print(f"Audit artifacts: {out}")


if __name__ == "__main__":
    main()
