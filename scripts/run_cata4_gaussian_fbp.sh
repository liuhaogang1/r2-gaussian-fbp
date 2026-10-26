#!/usr/bin/env bash
# Run from the repository root after uploading the updated Python files.
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u run_pipeline.py \
  --input_dir /root/fnvme/data_10_8/1/Cata4_8.4keV_500ms \
  --output_dir /root/fnvme/TEST_10.21/data_1_cata4_40nm_gaussian_fbp_auto \
  --model_path /root/fnvme/TEST_10.21/models/data_1_cata4_40nm_gaussian_fbp_auto \
  --config /root/fnvme/data_10_8/1/Cata4_8.4keV_500ms/config.txt \
  --projection_glob "Tomo_*.tiff" \
  --flat_glob "Flat_*.tiff" \
  --fbp_flat_field \
  --fbp_dark_value 0 \
  --fbp_flat_eps 1e-6 \
  --fbp_input_type transmission \
  --fbp_log_eps 1e-6 \
  --fbp_mode parallel \
  --fbp_coordinate_mode gaussian \
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
  --center_output /root/fnvme/TEST_10.21/data_1_cata4_40nm_gaussian_fbp_auto_center.json \
  --fbp_n_train 360 \
  --fbp_n_test 0 \
  --init_source train_fbp \
  --init_n_points 100000 \
  --init_density_thresh 0.10 \
  --init_density_rescale 0.25 \
  --init_normalize_percentile 99.5 \
  --init_visualize_fbp_nii \
  --train_config configs/pose_parallel.yaml \
  --train_pose_enabled \
  --train_max_num_gaussians 300000 \
  --test_iteration -1 \
  --test_skip_render_test \
  --seed 0 \
  "$@"
