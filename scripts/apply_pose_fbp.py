"""Optional offline pose transfer and FBP; never imported by the main pipeline."""

import argparse
from datetime import datetime, timezone
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
    FILTERS, backend_provenance, probe_tigre_full_geometry, reconstruct_full,
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
    return parser


def _save_volume(volume, scanner, output, name):
    volume = np.asarray(volume, dtype=np.float32)
    if volume.shape != tuple(scanner["nVoxel"]) or not np.isfinite(volume).all():
        raise ValueError(f"{name}: reconstruction must be finite XYZ with shape {scanner['nVoxel']}")
    atomic_npy(output / f"{name}.npy", volume)
    geometry = save_nifti_xyz(volume, scanner, output / f"{name}.nii.gz")
    return geometry


def _save_comparison(volumes, scanner, output):
    """Every volume and every plane shares ONE display window; raw data stays unchanged."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spacing, _ = volume_geometry(next(iter(volumes.values())).shape, scanner)
    planes = []
    for volume in volumes.values():
        x, y, z = (size // 2 for size in volume.shape)
        planes.append((volume[x], volume[:, y], volume[:, :, z]))
    values = np.concatenate([plane.ravel()[::max(1, plane.size // 100000)]
                             for row in planes for plane in row])
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        low, high = float(values.min()), float(values.max())
        if high <= low:
            high = low + 1
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
    return {"shared_display_window": [float(low), float(high)]}


def _input_unchanged(inputs):
    if (dataset_identity(inputs.dataset) != inputs.dataset_hash
            or file_identity(inputs.pose_path) != inputs.pose_identity):
        raise ValueError("Dataset or pose snapshot changed during export/reconstruction")


def run(args):
    if args.apply_beam_tilt and args.ablation:
        raise ValueError("--apply-beam-tilt and --ablation are mutually exclusive")
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
        if args.apply_beam_tilt and args.backend == "tigre":
            probe = probe_tigre_full_geometry()
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
        "angle_quadrature": "uniform_existing_TIGRE_normalization",
        "view_weights": [1.0] * len(inputs.records), "extra_spacing_weights": False,
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
            report["preview"] = _save_comparison(volumes, inputs.scanner, output / preview)
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
