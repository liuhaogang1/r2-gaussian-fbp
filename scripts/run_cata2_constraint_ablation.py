"""Run the Cata2 loss ablation from one shared denoised initialization."""

import argparse
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PurePosixPath("/root/fnvme/data_10_8/1/cata2_8.4kev_400ms_1")
OUTPUT_ROOT = PurePosixPath("/root/fnvme/TEST_11.17/data_1_cata2_40nm_1_v2_constraint_ablation_pair")
ITERATIONS = "30000"
SLICE_WEIGHTS = ("1", "0.5", "0.25", "0.1")
# Each group specifies its projection L1 weight, slice weights, and needle penalty.
GROUPS = (
    ("g1_slice", "0", ("1",), False),
    ("g2_l1", "1", (None,), False),
    ("g3_l1_slice", "1", SLICE_WEIGHTS, False),
    ("g4_l1_slice_needle", "1", SLICE_WEIGHTS, True),
    ("g5_slice_needle", "0", ("1",), True),
)
CUDA_ENV = {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def build_plan(output_root):
    dataset = output_root / "dataset"
    init_model = output_root / "shared_init"
    init_dir = init_model / "init_denoise"
    init_points = init_dir / "init_points.npy"
    references = {
        "raw_fbp": dataset / "vol_train_fbp.npy",
        "denoised_fbp": init_dir / "vol_init_denoised.npy",
    }
    common = [
        sys.executable, "-u", str(PROJECT_ROOT / "run_pipeline.py"),
        "--input_dir", str(INPUT_DIR), "--seed", "0", "--resume",
    ]
    prepare = [
        *common, "--start_from", "center", "--stop_after", "init",
        "--output_dir", str(dataset), "--model_path", str(init_model),
        "--config", str(INPUT_DIR / "config.txt"),
        "--center_method", "pair", "--center_pixel_size", "0.00004",
        "--center_output", str(output_root / "center_file" / "center.json"),
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
        "--fbp_n_train", "360", "--fbp_n_test", "0",
        "--init_source", "train_fbp", "--init_denoise", "--init_denoise_sigma", "0.8",
        "--init_denoise_save_nii", "--init_denoise_output_dir", str(init_dir),
        "--init_n_points", "100000", "--init_density_thresh", "0.1",
        "--init_density_rescale", "0.2", "--init_normalize_percentile", "99.5",
    ]
    training = [
        *common, "--start_from", "train", "--stop_after", "test",
        "--train_source_path", str(dataset), "--train_ply_path", str(init_points),
        "--train_config", str(PROJECT_ROOT / "configs" / "pose_parallel.yaml"),
        "--train_iterations", ITERATIONS, "--train_max_num_gaussians", "300000",
        "--no-train_pose_enabled", "--train_lambda_dssim", "0", "--train_lambda_tv", "0",
        "--train_densify_from_iter", "500", "--train_densify_until_iter", "15000",
        "--train_densification_interval", "100", "--train_densify_grad_threshold", "0.0015",
        "--train_densify_scale_threshold", "0.015",
        "--test_iteration", "-1", "--test_skip_render_test",
    ]
    slice_options = [
        "--train_reference_enabled", "--train_reference_method", "slice", "--train_reference_densify",
        "--train_reference_loss", "l1", "--train_reference_grad_weight", "0",
        "--train_reference_start_iter", "1", "--train_reference_end_iter", ITERATIONS,
        "--train_reference_interval", "1", "--train_reference_seed", "0",
        "--train_reference_slice_axis", "z", "--train_reference_slice_sample_ratio", "1.0",
    ]
    needle_options = [
        "--train_shape_enabled", "--train_shape_needle_enabled", "--no-train_shape_sheet_enabled",
        "--train_shape_needle_threshold", "1.1", "--train_lambda_shape_needle", "0.15",
        "--train_lambda_shape_sheet", "0", "--train_shape_hard_action", "none",
        "--train_shape_start_iter", "1", "--train_shape_end_iter", ITERATIONS,
        "--train_shape_ramp_iters", "0",
    ]

    jobs = []
    for round_name, reference in references.items():
        for group, projection_weight, weights, needle in GROUPS:
            for weight in weights:
                name = group
                options = ["--train_lambda_l1", projection_weight]
                if weight is None:
                    options += ["--no-train_reference_enabled", "--no-train_reference_densify"]
                else:
                    name += f"_w{weight.replace('.', 'p')}"
                    options += [*slice_options, "--train_reference_path", str(reference),
                                "--train_reference_weight", weight]
                options += needle_options if needle else ["--no-train_shape_enabled"]
                run_dir = output_root / "runs" / round_name / name
                command = [
                    *training, "--output_dir", str(run_dir),
                    "--model_path", str(output_root / "models" / round_name / name), *options,
                ]
                jobs.append((f"{round_name}/{name}", run_dir / "pipeline.log", command))
    return prepare, jobs, [dataset / "meta_data.json", init_points, *references.values()]


def run_job(name, log_path, command):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[START] {name} ({log_path})", flush=True)
    with log_path.open("ab") as log:
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, env={**os.environ, **CUDA_ENV},
            stdout=log, stderr=subprocess.STDOUT,
        )
    status = "OK" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"[{status}] {name}", flush=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--output-root", type=Path, help=f"Override the default output root ({OUTPUT_ROOT})")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve() if args.output_root else OUTPUT_ROOT
    if not args.dry_run:
        output_root = Path(output_root)
    prepare, jobs, shared_files = build_plan(output_root)
    prep_job = ("prepare", output_root / "prepare.log", prepare)
    if args.dry_run:
        env = [f"{key}={value}" for key, value in CUDA_ENV.items()]
        for name, _, command in [prep_job, *jobs]:
            print(f"[{name}]\n{shlex.join(['env', *env, *command])}")
        return 0

    import fcntl

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Batch already running in {output_root}", file=sys.stderr)
            return 1
        if not run_job(*prep_job):
            return 1
        missing = [str(path) for path in shared_files if not path.is_file()]
        if missing:
            print("Missing shared artifacts:\n" + "\n".join(missing), file=sys.stderr)
            return 1
        completed = sum(run_job(*job) for job in jobs)
        print(f"Completed {completed}/{len(jobs)} training jobs.")
        return int(completed != len(jobs))


if __name__ == "__main__":
    raise SystemExit(main())
