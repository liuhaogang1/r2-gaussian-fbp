"""Six shape ablations with explicit thresholds/weights and shared input data."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_pipeline as pipeline  # noqa: E402
from r2_gaussian.utils.artifact_utils import file_identity  # noqa: E402
from scripts import run_shape_grid as batch  # noqa: E402
from scripts.pipeline_artifacts import dataset_files  # noqa: E402


MANIFEST = "shape_ablation_manifest.json"
CASES = (
    ("A0_baseline", False, False, "none"),
    ("A1_needle", True, False, "none"),
    ("A2_sheet", False, True, "none"),
    ("A3_soft_both", True, True, "none"),
    ("A4_hard_repair", True, True, "repair"),
    ("A5_soft_repair", True, True, "repair"),
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name, help_text in (
        ("input_dir", "Raw Tomo/Flat TIFF directory"),
        ("config", "Acquisition config.txt"),
        ("output_dir", "NEW data/log/image root"),
        ("model_path", "NEW root for six independent models"),
    ):
        parser.add_argument("--" + name, type=Path, required=True, help=help_text)
    for name in ("needle_threshold", "sheet_threshold", "lambda_needle", "lambda_sheet"):
        parser.add_argument("--" + name, type=float, required=True)
    parser.add_argument("--shared_dataset_dir", type=Path, help="Existing compatible preprocessed dataset")
    parser.add_argument("--shared_init_path", type=Path, help="Existing denoised init_points.npy; pass both reuse options")
    parser.add_argument("--dry_run", action="store_true", help="Validate and print commands; no files written")
    parser.add_argument("--plot_only", action="store_true", help="Recreate six-case PNGs from completed volumes")
    parser.add_argument("--fail_fast", action="store_true", help="Stop after the first failed trial")
    # All remaining training settings are fixed to the previously agreed flow.
    parser.set_defaults(train_config=ROOT / "configs/pose_parallel.yaml")
    return parser


def validate_args(args):
    for name in ("needle_threshold", "sheet_threshold"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 1:
            raise ValueError(f"--{name} must be finite and greater than 1")
    for name in ("lambda_needle", "lambda_sheet"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and non-negative")
    if (args.shared_dataset_dir is None) != (args.shared_init_path is None):
        raise ValueError("Pass --shared_dataset_dir and --shared_init_path together")
    for name in ("shared_dataset_dir", "shared_init_path"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.plot_only:
        if args.dry_run:
            raise ValueError("--dry_run and --plot_only are mutually exclusive")
        for name in ("input_dir", "output_dir", "model_path", "config", "train_config"):
            setattr(args, name, getattr(args, name).expanduser().resolve())
        if not (args.output_dir / MANIFEST).is_file():
            raise ValueError(f"--plot_only requires an existing {MANIFEST}")
        return
    batch.validate_args(args)
    if args.shared_dataset_dir is not None:
        for protected in (args.shared_dataset_dir, args.shared_init_path.parent):
            for output in (args.output_dir, args.model_path):
                if output == protected or output in protected.parents or protected in output.parents:
                    raise ValueError("New output roots must be separate from reused input directories")


def original_options(args):
    options = batch.original_options(args)
    # Change only the four user-controlled shape parameters in the common
    # pipeline command; preprocessing and initialization do not consume them.
    for name, value in (
        ("--train_shape_needle_threshold", args.needle_threshold),
        ("--train_shape_sheet_threshold", args.sheet_threshold),
        ("--train_lambda_shape_needle", args.lambda_needle),
        ("--train_lambda_shape_sheet", args.lambda_sheet),
    ):
        options[options.index(name) + 1] = str(value)
    for name, value in (
        ("check_interval", 100), ("patience", 3), ("hard_needle_ratio", 50),
        ("hard_sheet_ratio", 50), ("max_hard_actions_per_check", 32),
        ("repair_max_density_gain", 4), ("low_density_threshold", 0),
        ("low_mass_threshold", 0), ("split_count", 2),
    ):
        options.extend(["--train_shape_" + name, str(value)])
    return options


def validate_reuse(args, expected):
    """Check prior pipeline parameters and content hashes without modifying it.

    Provenance deliberately uses saved artifact hashes rather than recomputing
    pipeline signatures: those include all Python sources, which can change
    when this new optional launcher is added to an existing grid experiment.
    """
    dataset, points = args.shared_dataset_dir, args.shared_init_path
    metadata_path = dataset / "meta_data.json"
    provenance_path = points.parent / "pipeline_manifest.json"
    report_path = points.parent / "denoise_report.json"
    for path in (metadata_path, points, provenance_path, report_path):
        if not path.is_file():
            raise ValueError(f"Cannot verify reused inputs; missing {path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    prior = provenance["resolved_arguments"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for stage in ("center", "fbp", "init"):
        if provenance.get("stage_status", {}).get(stage) not in ("success", "resumed"):
            raise ValueError(f"Reused pipeline has no verified completed {stage} stage")
    for key, path in (("input_dir", args.input_dir), ("fbp_config", args.config),
                      ("output_dir", dataset), ("init_output", points)):
        if prior.get(key) is None or Path(prior[key]).resolve() != path:
            raise ValueError(f"Reused inputs do not match {key}: expected {path}")
    # Paths may differ between batches, while all numerical preprocessing and
    # initialization settings must match the agreed original configuration.
    for key, value in vars(expected).items():
        if (not key.startswith(("center_", "fbp_", "init_"))
                or key in pipeline.PATH_KEYS or key == "fbp_sVoxel"):
            continue
        if prior.get(key) != value:
            raise ValueError(f"Reused parameter mismatch: {key}: {prior.get(key)!r} != {value!r}")
    for key in ("stripe_correction", "seed", "manual_off_detector", "center_json"):
        value = pipeline._jsonable(getattr(expected, key, None))
        if prior.get(key) != value:
            raise ValueError(f"Reused parameter mismatch: {key}")
    if prior.get("fbp_sVoxel_requested") != "auto":
        raise ValueError("Reused data must have been reconstructed with fbp_sVoxel=auto")
    scanner = metadata["scanner"]
    if list(scanner["nVoxel"]) != [256, 256, 256] or scanner["mode"] != "parallel":
        raise ValueError("Reused data must use the original parallel 256^3 grid")
    if not np.allclose(scanner["sVoxel"], prior["fbp_sVoxel"], rtol=0, atol=1e-12):
        raise ValueError("Reused scanner.sVoxel differs from its pipeline provenance")
    if len(metadata["proj_train"]) != 360 or len(metadata["proj_test"]) != 0:
        raise ValueError("Reused data must have the original 360/0 view split")

    checked = {}

    def verify(path, sha256):
        path = Path(path).resolve()
        if str(path) not in checked:
            checked[str(path)] = file_identity(path)
        if checked[str(path)]["sha256"] != sha256:
            raise ValueError(f"Reused artifact hash mismatch: {path}")

    recorded = set()
    for stage in ("center", "fbp", "init"):
        artifacts = provenance.get("stage_records", {}).get(stage, {}).get("artifacts", [])
        if not artifacts:
            raise ValueError(f"Missing {stage} artifact hashes in {provenance_path}")
        for artifact in artifacts:
            verify(artifact["path"], artifact["sha256"])
            recorded.add(Path(artifact["path"]).resolve())
    # A pipeline record cannot omit a projection/volume later used for training.
    for path in (*dataset_files(dataset), points, report_path):
        if path.resolve() not in recorded:
            raise ValueError(f"Reused training input is absent from saved artifact hashes: {path}")
    if Path(report["input"]["metadata_path"]).resolve() != metadata_path:
        raise ValueError("Reused initialization belongs to a different dataset")
    if Path(report["point_output"]).resolve() != points:
        raise ValueError("Reused denoise report belongs to a different initial point file")
    train_volume = Path(report["input"]["volume_path"]).resolve()
    if train_volume != dataset / "vol_train_fbp.npy":
        raise ValueError("Reused initialization must come from this dataset's train-view FBP")
    for path, digest in (
        (metadata_path, report["input"]["metadata_sha256"]),
        (train_volume, report["input"]["volume_sha256"]),
        (points, report["point_sha256"]),
        (report["volume_output"], report["volume_sha256"]),
    ):
        verify(path, digest)
    for key, value in {
        "sigma_voxels": 0.8, "normalization_percentile": 99.5,
        "density_threshold_normalized": 0.25, "density_rescale": 0.2,
        "seed": 0, "sampled_count": 100000, "shape_xyz": [256, 256, 256],
    }.items():
        if report.get(key) != value:
            raise ValueError(f"Reused denoise report mismatch: {key}")
    cloud = np.load(points, mmap_mode="r", allow_pickle=False)
    if cloud.shape != (100000, 4) or not np.isfinite(cloud).all() or np.any(cloud[:, 3] <= 0):
        raise ValueError("Reused point cloud must contain 100000 finite positive-density XYZ+density points")
    # The previous grid launcher also recorded the acquisition config digest.
    # Check it when present; the raw TIFFs are not reconstructed/reprocessed.
    config_hash_verified = False
    for filename in ("shape_grid_manifest.json", MANIFEST):
        outer = dataset.parent.parent / filename
        if outer.is_file():
            saved = json.loads(outer.read_text(encoding="utf-8"))
            for item in saved.get("configuration_identity", []):
                if Path(item["path"]).resolve() == args.config:
                    verify(args.config, item["sha256"])
                    config_hash_verified = True
    return {"provenance": str(provenance_path), "denoise_report": str(report_path),
            "configuration_hash_verified": config_hash_verified, "verified_files": list(checked.values())}


def build_plan(args):
    options = original_options(args)
    base = pipeline.resolve_args(pipeline.build_parser().parse_args(options))
    pipeline.validate_args(base)
    validation = None
    if args.shared_dataset_dir is not None:
        validation = validate_reuse(args, base)
        base.output_dir = args.shared_dataset_dir
        base.init_output = args.shared_init_path
        base.init_denoise_output_dir = args.shared_init_path.parent
    cases = []
    for name, needle, sheet, action in CASES:
        resolved = argparse.Namespace(**vars(base))
        resolved.model_path = args.model_path / name
        resolved.train_shape_enabled = name != "A0_baseline"
        resolved.train_shape_needle_enabled = needle
        resolved.train_shape_sheet_enabled = sheet
        resolved.train_lambda_shape_needle = args.lambda_needle if needle and name != "A4_hard_repair" else 0.0
        resolved.train_lambda_shape_sheet = args.lambda_sheet if sheet and name != "A4_hard_repair" else 0.0
        resolved.train_shape_hard_action = action
        pipeline.validate_args(resolved)
        cfg = args.output_dir / "cases" / name / "train_config.yml"
        final = resolved.train_iterations
        test_dir = resolved.model_path / "test" / f"iter_{final}"
        cases.append({
            "case": name, "iteration": final, "model_path": str(resolved.model_path), "config_path": str(cfg),
            "shape_enabled": resolved.train_shape_enabled, "needle_enabled": needle, "sheet_enabled": sheet,
            "needle_threshold": args.needle_threshold, "sheet_threshold": args.sheet_threshold,
            "lambda_needle": resolved.train_lambda_shape_needle, "lambda_sheet": resolved.train_lambda_shape_sheet,
            "hard_action": action, "volume_path": str(test_dir / "vol_pred.npy"),
            "pickle_path": str(resolved.model_path / "point_cloud" / f"iteration_{final}" / "point_cloud.pickle"),
            "train_command": batch.unbuffered(pipeline.build_train_command(resolved, base.init_output, cfg)),
            "test_command": batch.unbuffered(pipeline.build_test_command(resolved, None)),
            "resolved_arguments": pipeline._jsonable(vars(resolved)), "status": "pending", "error": "",
        })
    shared = None if validation else [sys.executable, "-u", str(ROOT / "run_pipeline.py"), *options, "--stop_after", "init"]
    return {"version": 1, "manifest_filename": MANIFEST, "experiment": "shape_ablation",
            "input_dir": str(args.input_dir), "output_dir": str(args.output_dir), "model_path": str(args.model_path),
            "requested_shape_parameters": {key: getattr(args, key) for key in (
                "needle_threshold", "sheet_threshold", "lambda_needle", "lambda_sheet")},
            "shared_command": shared, "shared_status": "pending", "shared_validation": validation,
            "metadata_path": str(base.output_dir / "meta_data.json"), "init_path": str(base.init_output),
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            "cases": cases}


def main(args):
    validate_args(args)
    if args.plot_only:
        manifest = json.loads((args.output_dir / MANIFEST).read_text(encoding="utf-8"))
        if Path(manifest["model_path"]) != args.model_path:
            raise ValueError("--model_path does not match the saved ablation")
        for key, value in manifest["requested_shape_parameters"].items():
            if getattr(args, key) != value:
                raise ValueError(f"--{key} does not match the saved experiment")
        if not any(case["status"] == "success" for case in manifest["cases"]):
            raise ValueError("No completed ablation cases available to plot")
        batch.plot_completed(args, manifest)
        batch.save_progress(args, manifest)
        return 0 if all(case["status"] == "success" for case in manifest["cases"]) else 1
    plan = build_plan(args)
    if args.dry_run:
        print("Six cases: " + ", ".join(case["case"] for case in plan["cases"]))
        print("Explicit shape parameters: " + json.dumps(plan["requested_shape_parameters"]))
        if plan["shared_validation"]:
            print(f"Verified {len(plan['shared_validation']['verified_files'])} reused input files")
        for case in plan["cases"]:
            print(f"[{case['case']}] needle={case['needle_enabled']} sheet={case['sheet_enabled']} "
                  f"lambda={case['lambda_needle']:g}/{case['lambda_sheet']:g} hard={case['hard_action']}")
        # execute_plan handles the common command listing and writes nothing.
    return batch.execute_plan(args, plan)


if __name__ == "__main__":
    try:
        sys.exit(main(build_parser().parse_args()))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
