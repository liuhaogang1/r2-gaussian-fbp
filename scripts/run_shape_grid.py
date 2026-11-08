"""Auto-pruning threshold grid and a final strong-constraint trial.

This optional wrapper calls existing pipeline/train/test implementations. It
does not change training defaults, shape losses, or random-seed behavior.
Previous repair/soft grids remain available as explicit experiment modes.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_pipeline as pipeline  # noqa: E402
from r2_gaussian.utils.artifact_utils import atomic_json, file_identity  # noqa: E402
from scripts.shape_grid_images import render_grid  # noqa: E402


THRESHOLDS = (5, 10, 20)
WEIGHTS = (0.01, 0.05, 0.1)
HARD_RATIOS = (12, 15, 20)
DENSITY_THRESHOLDS = (0.01, 0.03, 0.06)
MASS_THRESHOLDS = (1e-6, 5e-6, 2e-5)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input_dir", type=Path, required=True, help="Raw Tomo/Flat TIFF directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="NEW shared data/log/image root")
    parser.add_argument("--model_path", type=Path, required=True, help="NEW root for independent models")
    parser.add_argument("--config", type=Path, required=True, help="Acquisition config.txt")
    parser.add_argument("--train_config", type=Path, default=ROOT / "configs/pose_parallel.yaml")
    parser.add_argument("--experiment", choices=("auto_grid", "repair_ratios", "soft_grid"), default="auto_grid",
                        help="Default: nine auto density/mass combinations at H15, then one strong trial; previous grids remain available")
    parser.add_argument("--dry_run", action="store_true", help="Print the plan without writing/running anything")
    parser.add_argument("--plot_only", action="store_true", help="Recreate PNGs from this batch's completed volumes")
    parser.add_argument("--fail_fast", action="store_true", help="Stop after a failed trial; default continues")
    return parser


def validate_args(args):
    for key in ("input_dir", "output_dir", "model_path", "config", "train_config"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.dry_run and args.plot_only:
        raise ValueError("--dry_run and --plot_only are mutually exclusive")
    if args.output_dir == args.model_path:
        raise ValueError("--output_dir and --model_path must be different directories")
    for output in (args.output_dir, args.model_path):
        if output == args.input_dir or output in args.input_dir.parents or args.input_dir in output.parents:
            raise ValueError("Output/model directories must be separate from the raw input tree")
    if args.plot_only:
        if not (args.output_dir / "shape_grid_manifest.json").is_file():
            raise ValueError("--plot_only requires an existing shape_grid_manifest.json")
        return
    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory missing: {args.input_dir}")
    for path in (args.config, args.train_config):
        if not path.is_file():
            raise ValueError(f"Configuration missing: {path}")
    if not list(args.input_dir.glob("Tomo_*.tiff")) or not list(args.input_dir.glob("Flat_*.tiff")):
        raise ValueError("Input must contain Tomo_*.tiff and Flat_*.tiff")
    for output in (args.output_dir, args.model_path):
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f"Use a new/empty output root; existing results are never overwritten: {output}")


def original_options(args):
    """The user's original settings, with explicitly supplied experiment paths."""
    return [
        "--input_dir", str(args.input_dir),
        "--output_dir", str(args.output_dir / "shared/dataset"),
        "--model_path", str(args.model_path / "_shared"),
        "--config", str(args.config),
        "--projection_glob", "Tomo_*.tiff", "--flat_glob", "Flat_*.tiff",
        "--fbp_flat_field", "--fbp_dark_value", "0", "--fbp_flat_eps", "1e-6",
        "--stripe_correction", '{"snr":3.0,"la_size":61,"sm_size":21,"dim":1}',
        "--fbp_input_type", "transmission", "--fbp_log_eps", "1e-6",
        "--fbp_mode", "parallel", "--fbp_shift_v", "0",
        "--fbp_pixel_subsample", "1", "--fbp_recon_pixel_subsample", "1",
        "--fbp_resize_order", "1", "--fbp_projection_scale", "0.125",
        "--fbp_pixel_size", "0.00004", "--fbp_nVoxel", "256", "256", "256",
        "--fbp_sVoxel", "auto", "--fbp_offOrigin", "0", "0", "0",
        "--fbp_DSD", "7", "--fbp_DSO", "5", "--fbp_accuracy", "0.5", "--fbp_filter", "hann",
        "--center_method", "pair", "--center_pixel_size", "0.00004",
        "--center_output", str(args.output_dir / "shared/center.json"),
        "--fbp_n_train", "360", "--fbp_n_test", "0",
        "--init_source", "train_fbp", "--init_denoise", "--init_denoise_sigma", "0.8",
        "--init_denoise_save_nii",
        "--init_denoise_output_dir", str(args.output_dir / "shared/init"),
        "--init_output", str(args.output_dir / "shared/init/init_points.npy"),
        "--init_n_points", "100000", "--init_density_thresh", "0.25",
        "--init_density_rescale", "0.2", "--init_normalize_percentile", "99.5",
        "--train_config", str(args.train_config), "--train_pose_enabled",
        "--train_shape_enabled", "--train_shape_needle_enabled", "--train_shape_sheet_enabled",
        "--train_shape_needle_threshold", "10", "--train_shape_sheet_threshold", "10",
        "--train_lambda_shape_needle", "0.05", "--train_lambda_shape_sheet", "0.05",
        "--train_shape_start_iter", "500", "--train_shape_end_iter", "30000",
        "--train_shape_ramp_iters", "1000", "--train_shape_hard_action", "none",
        "--train_max_num_gaussians", "300000", "--train_iterations", "30000",
        "--test_iteration", "-1", "--test_skip_render_test", "--seed", "0",
    ]


def unbuffered(command):
    return [command[0], "-u", *command[1:]]


def trial_specs(experiment):
    if experiment == "repair_ratios":
        return [dict(case=f"H{ratio}_repair", threshold=10, weight=0.05, hard_ratio=ratio,
                     low_density_threshold=0, low_mass_threshold=0) for ratio in HARD_RATIOS]
    if experiment == "soft_grid":
        return [dict(case=f"T{threshold}_L{str(weight).replace('.', 'p')}", threshold=threshold,
                     weight=weight, hard_ratio=None) for threshold in THRESHOLDS for weight in WEIGHTS]
    if experiment != "auto_grid":
        raise ValueError(f"Unknown experiment: {experiment}")
    trials = []
    for density in DENSITY_THRESHOLDS:
        for mass in MASS_THRESHOLDS:
            index = len(trials) + 1
            density_tag = f"{density:g}".replace(".", "p")
            mass_tag = f"{mass:g}".replace(".", "p")
            trials.append(dict(
                case=f"E{index:02d}_H15_D{density_tag}_M{mass_tag}",
                threshold=10, weight=0.05, hard_ratio=15, trial_role="parameter_search",
                low_density_threshold=density, low_mass_threshold=mass,
            ))
    # This independent trial always runs AFTER all nine parameter combinations.
    trials.append(dict(
        case="E10_strong_T5_L0p1_H10", threshold=5, weight=0.1, hard_ratio=10,
        trial_role="strong_constraint", low_density_threshold=max(DENSITY_THRESHOLDS),
        low_mass_threshold=max(MASS_THRESHOLDS),
    ))
    return trials


def build_plan(args):
    options = original_options(args)
    if args.experiment in ("repair_ratios", "auto_grid"):
        options[options.index("--train_shape_hard_action") + 1] = (
            "auto" if args.experiment == "auto_grid" else "repair")
        for name, value in (
            ("check_interval", 100), ("patience", 3), ("max_hard_actions_per_check", 32),
            ("repair_max_density_gain", 4), ("low_density_threshold", 0), ("low_mass_threshold", 0),
        ):
            options.extend(["--train_shape_" + name, str(value)])
        if args.experiment == "auto_grid":
            options.extend(["--train_shape_hard_needle_ratio", "15", "--train_shape_hard_sheet_ratio", "15"])
    base = pipeline.resolve_args(pipeline.build_parser().parse_args(options))
    pipeline.validate_args(base)
    shared = [sys.executable, "-u", str(ROOT / "run_pipeline.py"), *options, "--stop_after", "init"]
    cases = []
    for spec in trial_specs(args.experiment):
        name, threshold, weight, hard_ratio = (spec[key] for key in ("case", "threshold", "weight", "hard_ratio"))
        resolved = argparse.Namespace(**vars(base))
        resolved.model_path = args.model_path / name
        resolved.train_shape_needle_threshold = threshold
        resolved.train_shape_sheet_threshold = threshold
        resolved.train_lambda_shape_needle = weight
        resolved.train_lambda_shape_sheet = weight
        if hard_ratio is not None:
            resolved.train_shape_hard_needle_ratio = hard_ratio
            resolved.train_shape_hard_sheet_ratio = hard_ratio
            resolved.train_shape_low_density_threshold = spec["low_density_threshold"]
            resolved.train_shape_low_mass_threshold = spec["low_mass_threshold"]
        pipeline.validate_args(resolved)
        config_path = args.output_dir / "cases" / name / "train_config.yml"
        final = resolved.train_iterations
        test_path = resolved.model_path / "test" / f"iter_{final}"
        case = {
            "case": name, "threshold": threshold, "weight": weight, "iteration": final,
            "model_path": str(resolved.model_path), "config_path": str(config_path),
            "volume_path": str(test_path / "vol_pred.npy"),
            "pickle_path": str(resolved.model_path / "point_cloud" / f"iteration_{final}" / "point_cloud.pickle"),
            "train_command": unbuffered(pipeline.build_train_command(resolved, base.init_output, config_path)),
            # test.py automatically binds the matching final pose snapshot.
            "test_command": unbuffered(pipeline.build_test_command(resolved, None)),
            "resolved_arguments": pipeline._jsonable(vars(resolved)),
            "status": "pending", "error": "",
        }
        if hard_ratio is not None:
            case.update(hard_ratio=hard_ratio, hard_action=resolved.train_shape_hard_action,
                        hard_needle_ratio=hard_ratio, hard_sheet_ratio=hard_ratio,
                        shape_enabled=True, needle_enabled=True, sheet_enabled=True,
                        needle_threshold=threshold, sheet_threshold=threshold,
                        lambda_needle=weight, lambda_sheet=weight,
                        low_density_threshold=resolved.train_shape_low_density_threshold,
                        low_mass_threshold=resolved.train_shape_low_mass_threshold)
        if "trial_role" in spec:
            case["trial_role"] = spec["trial_role"]
        cases.append(case)
    return {"version": 3, "experiment": args.experiment,
            "input_dir": str(args.input_dir), "output_dir": str(args.output_dir),
            "model_path": str(args.model_path), "shared_command": shared,
            "shared_status": "pending", "metadata_path": str(base.output_dir / "meta_data.json"),
            "init_path": str(base.init_output), "environment": {
                "CUDA_VISIBLE_DEVICES": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            "cases": cases}


def run_logged(command, log_path, environment):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + shlex.join(command), flush=True)
    print(f"  log: {log_path}", flush=True)
    env = dict(os.environ, **environment)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Process exited {result.returncode}; see {log_path}")


def shape_statistics(path):
    # Only read trusted point_cloud.pickle files generated by this batch.
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    raw = np.asarray(data["scale"], dtype=np.float64)
    bounds = data["scale_bound"]
    if bounds is None:
        scales = np.exp(raw)
    else:
        scales = np.exp(-np.logaddexp(0.0, -raw)) * (bounds[1] - bounds[0]) + bounds[0]
    density = np.logaddexp(0.0, np.asarray(data["density"], dtype=np.float64).reshape(-1))
    if (scales.ndim != 2 or scales.shape[1] != 3 or len(scales) == 0
            or density.shape != (len(scales),) or not np.isfinite(scales).all()
            or not np.isfinite(density).all() or np.any(scales <= 0)):
        raise ValueError(f"Invalid final Gaussian scales/density: {path}")
    ordered = np.sort(scales, axis=1)
    ratios = {"needle": ordered[:, 2] / ordered[:, 1], "sheet": ordered[:, 1] / ordered[:, 0]}
    mass = density * np.prod(scales, axis=1)
    stats = {"num_gaussians": len(scales)}
    for name, ratio in ratios.items():
        stats[f"{name}_p99"] = float(np.quantile(ratio, 0.99))
        for limit in (10, 12, 15, 20, 50):
            mask = ratio > limit
            stats[f"{name}_gt{limit}_fraction"] = float(mask.mean())
            stats[f"{name}_gt{limit}_mass_fraction"] = float(mass[mask].sum() / mass.sum()) if mass.sum() > 0 else 0.0
    return stats


def collect_metrics(case):
    result = shape_statistics(case["pickle_path"])
    test_dir = Path(case["volume_path"]).parent
    for filename, mapping in (
        ("eval2d_render_train.yml", {"psnr_2d": "train_psnr", "ssim_2d": "train_ssim"}),
        ("eval3d.yml", {"psnr_3d": "psnr_3d_vs_fbp", "ssim_3d": "ssim_3d_vs_fbp"}),
    ):
        values = yaml.safe_load((test_dir / filename).read_text(encoding="utf-8"))
        for source, target in mapping.items():
            value = float(values[source])
            # Perfect fits can produce inf PSNR; keep JSON valid and preserve
            # that result in the CSV instead of marking the trial as failed.
            result[target] = value if np.isfinite(value) else str(value)
    status = json.loads((test_dir / "evaluation_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed" or status.get("iteration") != case["iteration"]:
        raise ValueError("Final evaluation status/iteration mismatch")
    if status.get("n_test") != 0:
        raise ValueError("Expected the original n_test=0 split")
    return result


def save_progress(args, manifest):
    atomic_json(args.output_dir / manifest.get("manifest_filename", "shape_grid_manifest.json"), manifest)
    extra = ("shape_enabled", "needle_enabled", "sheet_enabled", "needle_threshold",
             "sheet_threshold", "lambda_needle", "lambda_sheet", "hard_action", "hard_ratio",
             "hard_needle_ratio", "hard_sheet_ratio", "low_density_threshold", "low_mass_threshold", "trial_role")
    extra = tuple(key for key in extra if any(key in case for case in manifest["cases"]))
    rows = [{key: case.get(key, "") for key in (
        "case", "threshold", "weight", "iteration", "status", "elapsed_seconds", "error", "volume_path", "png", *extra)}
        | case.get("metrics", {}) for case in manifest["cases"]]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_completed(args, manifest):
    completed = [case for case in manifest["cases"] if case["status"] == "success"]
    if not completed:
        return
    report = render_grid(completed, manifest["metadata_path"], args.output_dir / "images")
    for item in report["cases"]:
        next(case for case in completed if case["case"] == item["case"])["png"] = item["png"]
    total = len(manifest["cases"])
    manifest["plot_status"] = "completed" if len(completed) == total else "partial"
    manifest["plotted_cases"] = len(completed)
    print(f"Images: {args.output_dir / 'images'} ({len(completed)}/{total} cases, shared display window)", flush=True)


def main(args):
    validate_args(args)
    if args.plot_only:
        manifest = json.loads((args.output_dir / "shape_grid_manifest.json").read_text(encoding="utf-8"))
        if Path(manifest["model_path"]) != args.model_path:
            raise ValueError("--model_path does not match the saved batch")
        if not any(case["status"] == "success" for case in manifest["cases"]):
            raise ValueError("No completed cases available to plot")
        plot_completed(args, manifest)
        save_progress(args, manifest)
        return 0 if all(case["status"] == "success" for case in manifest["cases"]) else 1
    manifest = build_plan(args)
    return execute_plan(args, manifest)


def execute_plan(args, manifest):
    """Run a planned set of independent trials with fresh or reused inputs."""
    if args.dry_run:
        print("Environment: " + json.dumps(manifest["environment"]))
        if manifest["shared_command"]:
            print(shlex.join(manifest["shared_command"]))
        else:
            print(f"Reuse validated dataset: {Path(manifest['metadata_path']).parent}")
            print(f"Reuse validated initialization: {manifest['init_path']}")
        for case in manifest["cases"]:
            print(f"\n[{case['case']}] -> {case['model_path']}")
            print(shlex.join(case["train_command"]))
            print(shlex.join(case["test_command"]))
        print(f"{len(manifest['cases'])} cases; shared data/init; one XYZ x 25/50/75 PNG per case. No files written.")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.mkdir(parents=True, exist_ok=True)
    manifest["configuration_identity"] = [file_identity(args.config), file_identity(args.train_config)]
    save_progress(args, manifest)
    try:
        manifest["shared_status"] = "running"
        save_progress(args, manifest)
        if manifest["shared_command"]:
            run_logged(manifest["shared_command"], args.output_dir / "logs/shared.log", manifest["environment"])
        else:
            print(f"[REUSE] dataset: {Path(manifest['metadata_path']).parent}", flush=True)
            print(f"[REUSE] initialization: {manifest['init_path']}", flush=True)
        manifest["shared_identity"] = [file_identity(manifest["metadata_path"]), file_identity(manifest["init_path"])]
        manifest["shared_status"] = "success" if manifest["shared_command"] else "reused"
        save_progress(args, manifest)
    except (Exception, KeyboardInterrupt) as exc:
        manifest["shared_status"] = "failed"
        manifest["shared_error"] = f"{type(exc).__name__}: {exc}"
        save_progress(args, manifest)
        raise
    try:
        for case in manifest["cases"]:
            started = time.perf_counter()
            print(f"\n[START] {case['case']}", flush=True)
            case["status"] = "training"
            save_progress(args, manifest)
            try:
                resolved = argparse.Namespace(**case["resolved_arguments"])
                # Builder functions consume already resolved values; paths in
                # the manifest are strings, accepted by write_train_config.
                pipeline.write_train_config(resolved, Path(case["config_path"]),
                                            Path(manifest["metadata_path"]).parent,
                                            Path(case["model_path"]), Path(manifest["init_path"]))
                run_logged(case["train_command"], args.output_dir / "logs" / f"{case['case']}_train.log", manifest["environment"])
                if not Path(case["pickle_path"]).is_file():
                    raise FileNotFoundError(f"Final Gaussian model missing: {case['pickle_path']}")
                case["status"] = "testing"
                save_progress(args, manifest)
                run_logged(case["test_command"], args.output_dir / "logs" / f"{case['case']}_test.log", manifest["environment"])
                if not Path(case["volume_path"]).is_file():
                    raise FileNotFoundError(f"Final reconstruction missing: {case['volume_path']}")
                case["metrics"] = collect_metrics(case)
                case["status"] = "success"
            except Exception as exc:
                case["status"] = "failed"
                case["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                case["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                save_progress(args, manifest)
            print(f"[{case['status'].upper()}] {case['case']} {case['error']}", flush=True)
            if case["status"] == "failed" and args.fail_fast:
                break
    except KeyboardInterrupt:
        for case in manifest["cases"]:
            if case["status"] in ("training", "testing"):
                case.update(status="interrupted", error="Interrupted by user")
        raise
    finally:
        for case in manifest["cases"]:
            if case["status"] == "pending":
                case.update(status="skipped", error="Batch stopped before this trial")
        save_progress(args, manifest)
    try:
        plot_completed(args, manifest)
    except Exception as exc:
        manifest["plot_status"] = "failed"
        manifest["plot_error"] = f"{type(exc).__name__}: {exc}"
        print(f"Plotting failed: {exc}; use --plot_only after fixing the cause", file=sys.stderr)
    save_progress(args, manifest)
    return 0 if all(case["status"] == "success" for case in manifest["cases"]) and manifest.get("plot_status") == "completed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main(build_parser().parse_args()))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
