#!/usr/bin/env bash
# Run sequentially on GPU 0. Each dataset gets a separate Python process.
set -euo pipefail
trap 'exit 130' INT
trap 'exit 143' TERM

INPUT_ROOT="${INPUT_ROOT:-/root/fnvme/data_10_8/1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/fnvme/TEST_10.28}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/fnvme/r2-gaussian-fbp}"

cd "$PROJECT_ROOT"
[[ -d "$INPUT_ROOT" ]] || { echo "Input directory missing: $INPUT_ROOT" >&2; exit 1; }
[[ -f run_pipeline.py && -f configs/pose_parallel.yaml ]] || {
  echo "Pipeline or training configuration missing in: $PROJECT_ROOT" >&2
  exit 1
}
command -v python >/dev/null

LOG_ROOT="$OUTPUT_ROOT/logs/batch_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUTPUT_ROOT/models" "$OUTPUT_ROOT/center_file" "$LOG_ROOT"
SUMMARY="$LOG_ROOT/summary.tsv"
printf 'dataset\tstatus\texit_code\tlog\n' > "$SUMMARY"

shopt -s nullglob
datasets=("$INPUT_ROOT"/*/)
if (( ${#datasets[@]} == 0 )); then
  echo "No dataset directories found: $INPUT_ROOT" >&2
  exit 1
fi

for input_dir in "${datasets[@]}"; do
  input_dir="${input_dir%/}"
  name="${input_dir##*/}"
  tag="data_1_${name}"
  log_file="$LOG_ROOT/$tag.log"

  if [[ ! -f "$input_dir/config.txt" ]]; then
    echo "[SKIP] $name: config.txt missing"
    printf '%s\tSKIPPED_NO_CONFIG\t-\t-\n' "$name" >> "$SUMMARY"
    continue
  fi

  printf '\n[START] %s | %s\n' "$name" "$(date '+%F %T')"
  if CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u run_pipeline.py \
      --input_dir "$input_dir" \
      --output_dir "$OUTPUT_ROOT/$tag" \
      --model_path "$OUTPUT_ROOT/models/$tag" \
      --config "$input_dir/config.txt" \
      --projection_glob "Tomo_*.tiff" \
      --flat_glob "Flat_*.tiff" \
      --fbp_flat_field \
      --fbp_dark_value 0 \
      --fbp_flat_eps 1e-6 \
      --stripe_correction '{"snr":3.0,"la_size":61,"sm_size":21,"dim":1}' \
      --fbp_input_type transmission \
      --fbp_log_eps 1e-6 \
      --fbp_mode parallel \
      --fbp_shift_v 0 \
      --fbp_pixel_subsample 1 \
      --fbp_recon_pixel_subsample 1 \
      --fbp_resize_order 1 \
      --fbp_projection_scale 0.125 \
      --fbp_pixel_size 0.00004 \
      --fbp_nVoxel 256 256 256 \
      --fbp_sVoxel auto \
      --fbp_offOrigin 0 0 0 \
      --fbp_DSD 7 \
      --fbp_DSO 5 \
      --fbp_accuracy 0.5 \
      --fbp_filter hann \
      --center_method pair \
      --center_pixel_size 0.00004 \
      --center_output "$OUTPUT_ROOT/center_file/$tag.json" \
      --fbp_n_train 360 \
      --fbp_n_test 0 \
      --init_source train_fbp \
      --init_n_points 100000 \
      --init_density_thresh 0.10 \
      --init_density_rescale 0.25 \
      --init_normalize_percentile 99.5 \
      --train_config "$PROJECT_ROOT/configs/pose_parallel.yaml" \
      --train_pose_enabled \
      --train_max_num_gaussians 300000 \
      --test_iteration -1 \
      --test_skip_render_test \
      --seed 0 2>&1 | tee "$log_file"; then
    echo "[OK] $name"
    printf '%s\tSUCCESS\t0\t%s\n' "$name" "$log_file" >> "$SUMMARY"
  else
    # Capture immediately: the next command would overwrite PIPESTATUS.
    stage_codes=("${PIPESTATUS[@]}")
    echo "[FAILED] $name (python=${stage_codes[0]}, tee=${stage_codes[1]}). Continuing."
    printf '%s\tFAILED\t%s\t%s\n' "$name" "${stage_codes[0]}" "$log_file" >> "$SUMMARY"
  fi
done

printf '\nBatch complete. Summary: %s\n' "$SUMMARY"
cat "$SUMMARY"
