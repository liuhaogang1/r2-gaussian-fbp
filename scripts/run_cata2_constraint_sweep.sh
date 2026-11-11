#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: bash %s [--dry-run] [slice|block|ssim [0.1|0.2|0.3|0.4|0.5]]\n' "$0"
}

dry_run=false
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  --dry-run) dry_run=true; shift ;;
esac
if (( $# > 2 )); then
  usage >&2
  exit 2
fi
groups=(slice block ssim)
weights=(0.1 0.2 0.3 0.4 0.5)
if (( $# >= 1 )); then
  case "$1" in
    slice|block|ssim) groups=("$1") ;;
    *) usage >&2; exit 2 ;;
  esac
fi
if (( $# == 2 )); then
  case "$2" in
    0.1|0.2|0.3|0.4|0.5) weights=("$2") ;;
    *) usage >&2; exit 2 ;;
  esac
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
input_dir="${INPUT_DIR:-/root/fnvme/data_10_8/1/cata2_8.4kev_400ms_1}"
sweep_root="${SWEEP_ROOT:-/root/fnvme/TEST_11.11/data_1_cata2_40nm_1_v2_constraint_sweep}"
python_bin="${PYTHON_BIN:-python}"
dataset_dir="$sweep_root/dataset"
init_dir="$sweep_root/shared_init"
iterations=30000
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

common=(
  --input_dir "$input_dir"
  --seed 0
  --resume
)
preparation=(
  --output_dir "$dataset_dir"
  --model_path "$init_dir"
  --stop_after init
  --config "$input_dir/config.txt"
  --projection_glob 'Tomo_*.tiff'
  --flat_glob 'Flat_*.tiff'
  --fbp_flat_field
  --fbp_dark_value 0
  --fbp_flat_eps 1e-6
  --stripe_correction '{"snr":3.0,"la_size":61,"sm_size":21,"dim":1}'
  --fbp_input_type transmission
  --fbp_log_eps 1e-6
  --fbp_mode parallel
  --fbp_shift_v 0
  --fbp_pixel_subsample 1
  --fbp_recon_pixel_subsample 1
  --fbp_resize_order 1
  --fbp_projection_scale 0.125
  --fbp_pixel_size 0.00004
  --fbp_nVoxel 256 256 256
  --fbp_sVoxel auto
  --fbp_offOrigin 0 0 0
  --fbp_DSD 7
  --fbp_DSO 5
  --fbp_accuracy 0.5
  --fbp_filter hann
  --center_method pair
  --center_pixel_size 0.00004
  --center_output "$sweep_root/center.json"
  --fbp_n_train 360
  --fbp_n_test 0
  --init_source train_fbp
  --init_denoise
  --init_denoise_sigma 0.8
  --init_denoise_save_nii
  --init_denoise_output_dir "$init_dir"
  --init_n_points 100000
  --init_density_thresh 0.25
  --init_density_rescale 0.2
  --init_normalize_percentile 99.5
)
training=(
  --start_from train
  --train_source_path "$dataset_dir"
  --train_ply_path "$init_dir/init_points.npy"
  --train_config "$project_root/configs/pose_parallel.yaml"
  --train_iterations "$iterations"
  --no-train_pose_enabled
  --no-train_shape_enabled
  --train_max_num_gaussians 300000
  --test_iteration "$iterations"
  --test_skip_render_test
)
reference=(
  --train_reference_enabled
  --train_reference_path "$dataset_dir/vol_train_fbp.npy"
  --train_reference_loss l1_grad
  --train_reference_grad_weight 0.1
  --train_reference_start_iter 500
  --train_reference_end_iter "$iterations"
  --train_reference_interval 50
  --train_reference_seed 0
  --train_lambda_dssim 0.25
)

run_logged() {
  local name="$1" log_file="$2"
  shift 2
  if "$dry_run"; then
    printf '\n[%s]\n' "$name"
    printf '%q ' env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
      "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF" "$@"
    printf '\n'
    return
  fi

  printf '\n[START] %s\n' "$name"
  "$@" 2>&1 | tee "$log_file"
}

batch_id="$(date +%Y%m%d_%H%M%S)_$$"
log_dir="$sweep_root/logs/$batch_id"
if ! "$dry_run"; then
  command -v "$python_bin" >/dev/null
  if [[ ! -f "$input_dir/config.txt" ]]; then
    printf 'Missing input config: %s\n' "$input_dir/config.txt" >&2
    exit 1
  fi
  mkdir -p -- "$log_dir"
  # Keep simultaneous invocations from writing the same dataset or model.
  exec 9>"$sweep_root/.lock"
  flock -n 9 || { printf 'Sweep is locked: %s\n' "$sweep_root" >&2; exit 1; }
  summary="$log_dir/summary.tsv"
  printf 'group\tweight\tstatus\tmodel_path\tlog_path\n' > "$summary"
fi

# The preparation manifest covers the shared dataset and the denoised point cloud.
run_logged prepare "$log_dir/prepare.log" \
  "$python_bin" -u run_pipeline.py "${common[@]}" "${preparation[@]}"

failed=0
for group in "${groups[@]}"; do
  for weight in "${weights[@]}"; do
    name="${group}_w${weight}"
    model_dir="$sweep_root/models/$name"
    run_dir="$sweep_root/runs/$name"
    run_log="$run_dir/$batch_id.log"
    case "$group" in
      slice)
        options=("${reference[@]}" --train_reference_method slice
          --train_reference_weight "$weight"
          --train_reference_slice_axis z --train_reference_slice_sample_ratio 1.0)
        ;;
      block)
        options=("${reference[@]}" --train_reference_method block
          --train_reference_weight "$weight"
          --train_reference_block_size 32 32 32 --train_reference_block_stride 16 16 16
          --train_reference_block_sampling density_mix --train_reference_block_density_mix 0.7)
        ;;
      ssim)
        options=(--no-train_reference_enabled --train_lambda_dssim "$weight")
        ;;
    esac
    if ! "$dry_run"; then
      mkdir -p -- "$run_dir"
    fi
    if run_logged "$name" "$run_log" "$python_bin" -u run_pipeline.py \
        "${common[@]}" --output_dir "$run_dir" --model_path "$model_dir" \
        "${training[@]}" "${options[@]}"; then
      status=SUCCESS
    else
      status=FAILED
      failed=1
      printf '[FAILED] %s\n' "$name" >&2
    fi
    if ! "$dry_run"; then
      printf '%s\t%s\t%s\t%s\t%s\n' "$group" "$weight" "$status" "$model_dir" "$run_log" >> "$summary"
    fi
  done
done

if ! "$dry_run"; then
  printf '\nSummary: %s\n' "$summary"
fi

if (( failed == 0 && ${#groups[@]} == 3 && ${#weights[@]} == 5 )); then
  run_logged montage "$log_dir/montage.log" "$python_bin" -u scripts/plot_cata2_constraint_sweep.py \
    "$sweep_root" --iteration "$iterations"
fi
exit "$failed"
