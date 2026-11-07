"""Optional offline pose transfer and FBP; never imported by the main pipeline."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2_gaussian.utils.artifact_utils import (  # noqa: E402
    atomic_json, atomic_npy, atomic_npz, dataset_identity, file_identity,
)
from r2_gaussian.utils.pose_fbp import (  # noqa: E402
    export_projections, load_inputs, load_projection, read_exported_projections,
)
from r2_gaussian.utils.pose_fbp_geometry import build_geometry, load_geometry  # noqa: E402
from r2_gaussian.utils.pose_fbp_recon import (  # noqa: E402
    FILTERS, backend_provenance, probe_tigre_full_geometry, probe_tigre_standard_geometry, reconstruct_full,
    reconstruct_standard, require_backend, software_versions,
)
from r2_gaussian.utils.volume_export_utils import save_nifti_xyz, volume_geometry  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared dataset with meta_data.json and training projections")
    parser.add_argument("--pose", type=Path, required=True, help="Bound iteration_<N>/pose_offsets.npz (parallel format 2)")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory; input datasets are never overwritten")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply-beam-tilt", action="store_true", help="Full 3D pose FBP; applies beam tilt in geometry")
    modes.add_argument("--ablation", action="store_true", help="Partial mode only: original, shifts, shifts+roll, shifts+roll+angles")
    parser.add_argument("--backend", choices=("tigre", "cpu"), default="tigre",
                        help="Default: TIGRE GPU. CPU is an explicitly selected small-volume reference")
    parser.add_argument("--filter", choices=FILTERS, default="ram_lak",
                        help="Explicit effective filter shared by every comparison; default matches upstream TIGRE FBP")
    parser.add_argument("--export-only", action="store_true", help="Save projections and full geometry if selected; no reconstruction dependency required")
    parser.add_argument("--comparison-volume", type=Path,
                        help="Full mode: existing XYZ FBP for blinking comparison; defaults to <dataset>/vol_train_fbp.npy")
    return parser


def _save_volume(volume, scanner, output, name):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.shape != tuple(scanner["nVoxel"]) or not np.isfinite(volume).all():
        raise ValueError(f"{name}: reconstruction must be finite XYZ with shape {scanner['nVoxel']}")
    atomic_npy(output / f"{name}.npy", volume)
    geometry = save_nifti_xyz(volume, scanner, output / f"{name}.nii.gz")
    return geometry


def _display_window(reference):
    # Same rule as prepare_fbp_tiff.save_fbp_preview(): percentiles of the
    # positive volume, so zero/negative background is black. Comparisons use
    # the FIRST volume's window for both images, never independent contrast.
    positive = reference[np.isfinite(reference) & (reference > 0)]
    if positive.size:
        low, high = np.percentile(positive, [1, 99.5])
        if high <= low:
            low, high = float(positive.min()), float(positive.max())
        if high <= low:
            high = low + 1
    else:
        low, high = 0.0, 1.0
    del positive
    return float(low), float(high)


def _save_comparison(volumes, scanner, output):
    """Match prepare_fbp_tiff previews, sharing the reference's positive window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_name, reference = next(iter(volumes.items()))
    low, high = _display_window(reference)
    spacing, _ = volume_geometry(reference.shape, scanner)
    planes = []
    for volume in volumes.values():
        x, y, z = (size // 2 for size in volume.shape)
        planes.append((volume[x], volume[:, y], volume[:, :, z]))
    fig, axes = plt.subplots(len(volumes), 3, figsize=(13, 4 * len(volumes)),
                             squeeze=False, constrained_layout=True)
    try:
        for row, (name, volume), row_planes in zip(axes, volumes.items(), planes):
            for axis, plane, normal, horizontal, vertical in zip(row, row_planes, "XYZ", (1, 0, 0), (2, 2, 1)):
                image = axis.imshow(plane.T, origin="lower", cmap="gray", vmin=low, vmax=high,
                                    extent=(0, plane.shape[0] * spacing[horizontal],
                                            0, plane.shape[1] * spacing[vertical]), aspect="equal")
                axis.set_title(f"{name} | {normal} center")
                axis.set_xlabel(f"{'XYZ'[horizontal]} (dataset units, local)")
                axis.set_ylabel(f"{'XYZ'[vertical]} (dataset units, local)")
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Reconstruction value")
        with output.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=140)
    finally:
        plt.close(fig)
    return {"shared_display_window": [float(low), float(high)],
            "window_reference": reference_name, "window_rule": "positive_volume_percentiles_1_99.5"}


def _load_comparison_reference(inputs, requested):
    """Read an existing train-view FBP, without starting another reconstruction."""
    path = (requested if requested is not None else inputs.dataset / "vol_train_fbp.npy").resolve()
    info = {"path": str(path), "selection": "explicit" if requested is not None else "dataset_train_fbp",
            "additional_reconstruction": False,
            "note": "Existing FBP retains its original filtering/normalization; this is not an isolated pose ablation."}
    volume = None
    try:
        reference_identity = file_identity(path)
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(volume, np.ndarray):
            volume.close()
            volume = None
            raise ValueError("Comparison volume must be a single .npy array")
        if (volume.shape != tuple(inputs.scanner["nVoxel"])
                or not np.issubdtype(volume.dtype, np.floating) or not np.isfinite(volume).all()):
            raise ValueError("Comparison volume must be finite floating-point XYZ on the dataset nVoxel grid")
        sidecars = [candidate for suffix in (".cache.json", ".stripe.json")
                    if (candidate := path.with_suffix(suffix)).exists()]
        if sidecars:
            bound = False
            for sidecar in sidecars:
                try:
                    cached = json.loads(sidecar.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if (isinstance(cached, dict) and isinstance(cached.get("signature"), dict)
                        and isinstance(cached.get("artifact"), dict)
                        and cached["signature"].get("dataset") == inputs.dataset_hash
                        and cached["artifact"].get("sha256") == reference_identity["sha256"]):
                    bound = True
                    info["cache_sidecar"] = file_identity(sidecar)
                    break
            if not bound:
                raise ValueError("Comparison FBP cache does not match the training dataset or volume content")
            info["binding"] = "verified_dataset_and_volume_hash"
        else:
            # Older volumes have no sidecar. State that view/geometry provenance
            # is unverified; never silently substitute an all-view or denoised FBP.
            info["binding"] = "unverified_no_cache_sidecar"
        if file_identity(path) != reference_identity:
            raise ValueError("Comparison volume changed while being loaded")
        info["identity"] = reference_identity
        return volume, info
    except (OSError, ValueError) as error:
        if volume is not None and getattr(volume, "_mmap", None) is not None:
            volume._mmap.close()
        if requested is not None:
            raise ValueError(f"Cannot use --comparison-volume {path}: {error}") from error
        info.update(status="skipped", reason=str(error))
        return None, info


def _save_blink_comparison(reference, corrected, output, display):
    """Fixed grayscale palette, shared window and identical XYZ slice indices."""
    from PIL import Image, ImageDraw

    low, high = display["shared_display_window"]
    palette = [value for value in range(256) for _ in range(3)]
    files = {}
    for axis, normal in enumerate("xyz"):
        index = reference.shape[axis] // 2
        frames = []
        for volume, label in ((reference, "Original FBP"), (corrected, "Pose-corrected FBP")):
            plane = np.take(volume, index, axis=axis)
            pixels = np.round(np.flipud(np.clip((plane - low) / (high - low), 0, 1).T) * 255).astype(np.uint8)
            image = Image.fromarray(pixels)
            scale = 2 if max(image.size) <= 256 else 1
            image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
            frame = Image.new("L", (max(256, image.width), image.height + 44), 0)
            draw = ImageDraw.Draw(frame)
            draw.text((8, 4), label, fill=255)
            draw.text((8, 23), f"{normal.upper()} center | index {index}", fill=255)
            frame.paste(image, ((frame.width - image.width) // 2, 44))
            frame = frame.convert("P")
            frame.putpalette(palette)
            frames.append(frame)
        filename = f"blink_center_{normal}.gif"
        with (output / filename).open("xb") as stream:
            frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:],
                           duration=900, loop=0, optimize=False, disposal=2)
        files[normal] = filename
    return {"status": "written", "files": files, "frame_duration_ms": 900, "loop": "forever",
            "frame_order": ["original_fbp", "pose_corrected_fbp"],
            "center_indices_xyz": [size // 2 for size in reference.shape], **display,
            "orientation": "flipud(plane.T), matching prepare_fbp_tiff PNGs",
            "volume_data_modified": False, "registration_applied": False}


def _input_unchanged(inputs):
    if (dataset_identity(inputs.dataset) != inputs.dataset_hash
            or file_identity(inputs.pose_path) != inputs.pose_identity):
        raise ValueError("Dataset or pose snapshot changed during export/reconstruction")


def run(args):
    if args.apply_beam_tilt and args.ablation:
        raise ValueError("--apply-beam-tilt and --ablation are mutually exclusive")
    if args.comparison_volume is not None and not args.apply_beam_tilt:
        raise ValueError("--comparison-volume requires --apply-beam-tilt; partial mode already reconstructs its original baseline")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory must not already exist: {output}")
    dataset = args.dataset.resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("Output directory must be outside the input dataset")
    inputs = load_inputs(dataset, args.pose, args.iteration)
    if output == inputs.pose_path.parent or inputs.pose_path.parent in output.parents:
        raise ValueError("Output directory must be outside the input pose snapshot")
    mode = "full_pose" if args.apply_beam_tilt else "partial"
    corrected_angles = inputs.angles + inputs.pose["delta_angle"]
    full_geometry = build_geometry(inputs.scanner, corrected_angles, inputs.pose["beam_tilt"]) if args.apply_beam_tilt else None
    probe = {"status": "not_run_export_only"}
    if not args.export_only:
        require_backend(args.backend)
        if args.backend == "tigre":
            probe = probe_tigre_full_geometry() if args.apply_beam_tilt else probe_tigre_standard_geometry()
        else:
            probe = {"status": "cpu_reference" if args.backend == "cpu" else "standard_path"}
    report = {
        "format_version": 1, "status": "exporting", "mode": mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(inputs.dataset), "dataset_identity": inputs.dataset_hash,
        "pose": inputs.pose_identity, "model_iteration": inputs.iteration,
        "n_views": len(inputs.records), "scanner": inputs.scanner,
        "beam_tilt_rad": inputs.pose["beam_tilt"].tolist(),
        "beam_tilt_application": "3d_geometry" if args.apply_beam_tilt else "ignored",
        "applied_parameters": ["delta_angle", "global_shift", "view_shift", "detector_roll"]
                              + (["beam_tilt"] if args.apply_beam_tilt else []),
        "ignored_parameters": [] if args.apply_beam_tilt else ["beam_tilt"],
        "parameter_application": {"global_shift": "pixels", "view_shift": "pixels",
                                  "detector_roll": "pixels", "delta_angle": "geometry",
                                  "beam_tilt": "geometry" if args.apply_beam_tilt else "ignored"},
        "interpolation": {"order": 1, "mode": "constant", "fill": 0, "mask": "supported_pixel_center_domain"},
        "effective_filter": args.filter, "scanner_configured_filter": inputs.scanner.get("filter"),
        "angle_quadrature": "uniform_pi_over_n",
        "view_weights": [float(np.pi / len(inputs.records))] * len(inputs.records), "extra_spacing_weights": False,
        "filter_implementation": "shared_physical_frequency_ramp_zero_dc",
        "backend": args.backend, "backend_probe": probe, "ablation": args.ablation,
        "export_only": args.export_only, "software": software_versions(),
        "implementation": [file_identity(ROOT / path) for path in (
            "scripts/apply_pose_fbp.py", "r2_gaussian/utils/pose_fbp.py",
            "r2_gaussian/utils/pose_fbp_geometry.py", "r2_gaussian/utils/pose_fbp_recon.py")],
        "reconstruction_note": (
            "Full ray geometry; directional-ramp uniform-quadrature FBP approximation for the saved trajectory. "
            "Finite angular coverage/missing frequencies are not recovered."
            if args.apply_beam_tilt else "Partial transfer: beam tilt is ignored even when nonzero."),
    }
    started = time.monotonic()
    volumes = {}
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "run_report.json", report)
    try:
        print(f"Mode: {mode}; beam tilt: {report['beam_tilt_application']} ({report['beam_tilt_rad']} rad)", flush=True)
        views = export_projections(inputs, output, beam_tilt_in_geometry=args.apply_beam_tilt)
        atomic_npy(output / "angles_original.npy", inputs.angles)
        report["valid_fraction_range"] = [min(view["valid_fraction"] for view in views),
                                           max(view["valid_fraction"] for view in views)]
        report["projection_manifest"] = file_identity(output / "views.json")
        if full_geometry is not None:
            atomic_npz(output / "geometry_full_pose.npz", **full_geometry)
            report["full_geometry"] = file_identity(output / "geometry_full_pose.npz")
        variants = []
        if args.ablation:
            for name, apply_roll in (("shift_only", False), ("shift_roll", True)):
                target = output / "ablation" / name
                export_projections(inputs, target, roll=apply_roll, angles=False)
                variants.append((name, target))
        _input_unchanged(inputs)
        if args.export_only:
            report["status"] = "exported"
            report["reconstruction_performed"] = False
        else:
            report["status"] = "reconstructing"
            report["backend_provenance"] = backend_provenance(args.backend)
            atomic_json(output / "run_report.json", report)
            report["volumes"] = {}

            def record_volume(name, volume):
                report["volumes"][name] = _save_volume(volume, inputs.scanner, output, name)
                # Retain only file-backed arrays for the shared-window preview.
                # Ablation on large grids must not retain four full RAM volumes.
                volumes[name] = np.load(output / f"{name}.npy", mmap_mode="r", allow_pickle=False)

            if args.apply_beam_tilt:
                # Both projections AND geometry are reloaded from disk.
                projections, angles, _ = read_exported_projections(output)
                saved_geometry = load_geometry(output / "geometry_full_pose.npz")
                if not np.array_equal(angles, saved_geometry["scan_angles"]):
                    raise ValueError("Saved full geometry does not match saved projection angles")
                if (not np.array_equal(saved_geometry["nDetector"], inputs.scanner["nDetector"])
                        or not np.allclose(saved_geometry["pixel_size_vu"], inputs.scanner["dDetector"], rtol=0, atol=0)):
                    raise ValueError("Saved full geometry does not match the scanner detector")
                record_volume("fbp_full_pose", reconstruct_full(projections, saved_geometry, inputs.scanner,
                                                                args.filter, args.backend))
                del projections
            else:
                projections = np.stack([load_projection(inputs.dataset / record["file_path"], inputs.scanner["nDetector"])
                                        for record in inputs.records])
                _input_unchanged(inputs)
                print("Reconstruct original baseline", flush=True)
                record_volume("fbp_original", reconstruct_standard(projections, inputs.angles, inputs.scanner, args.filter, args.backend))
                del projections
                for name, target in variants + [("partial", output)]:
                    print(f"Reconstruct {name} from saved projections", flush=True)
                    projections, angles, _ = read_exported_projections(target)
                    record_volume(f"fbp_{name}", reconstruct_standard(projections, angles, inputs.scanner, args.filter, args.backend))
                    del projections
            preview = "fbp_full_pose_slices.png" if args.apply_beam_tilt else "comparison_xyz.png"
            if args.apply_beam_tilt:
                reference, reference_info = _load_comparison_reference(inputs, args.comparison_volume)
                if reference is not None:
                    # Include the existing reference in the preview only; the
                    # volumes report still lists just newly reconstructed data.
                    volumes["fbp_existing_reference"] = reference
                    preview_volumes = {"fbp_existing_reference": reference, "fbp_full_pose": volumes["fbp_full_pose"]}
                else:
                    preview_volumes = volumes
                corrected = volumes["fbp_full_pose"]
            else:
                reference = volumes["fbp_original"]
                reference_info = {"selection": "same_run_original", "additional_reconstruction": False,
                                  "identity": file_identity(output / "fbp_original.npy")}
                corrected = volumes["fbp_partial"]
                preview_volumes = volumes
            report["preview"] = _save_comparison(preview_volumes, inputs.scanner, output / preview)
            if reference is not None:
                report["blink_comparison"] = _save_blink_comparison(reference, corrected, output, report["preview"])
                report["blink_comparison"]["reference"] = reference_info
                if file_identity(reference_info["identity"]["path"]) != reference_info["identity"]:
                    raise ValueError("Comparison volume changed during preview generation")
                print("Saved blink_center_x.gif, blink_center_y.gif, blink_center_z.gif", flush=True)
            else:
                report["blink_comparison"] = reference_info
                print(f"Blink comparison skipped: {reference_info['reason']}", flush=True)
            report["status"] = "complete"
            report["reconstruction_performed"] = True
        _input_unchanged(inputs)
        print(f"{report['status']}: {output}", flush=True)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # numpy.memmap otherwise releases handles only at garbage collection;
        # matplotlib may retain views longer, which locks .npy files on Windows.
        for volume in volumes.values():
            volume._mmap.close()
        volumes.clear()
        report["elapsed_seconds"] = time.monotonic() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "run_report.json", report)
    return report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"pose-fbp: {error}\n")


if __name__ == "__main__":
    main()
