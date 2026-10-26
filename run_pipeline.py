"""Run every project stage through one reproducible command.

The stage implementations remain in their original files. This module is
the only public orchestration entry point: it normalizes command-line/YAML
configuration, validates dependencies, builds each stage command, and records
the resolved run in pipeline_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
STAGES = ("center", "fbp", "init", "train", "test")


DEFAULTS: dict[str, Any] = {
    # Global orchestration
    "input_dir": None,
    "output_dir": None,
    "model_path": None,
    "pipeline_config": None,
    "fbp_config": None,
    "train_config": None,
    "seed": 0,
    "skip_center": False,
    "skip_fbp": False,
    "skip_init": False,
    "skip_train": False,
    "skip_test": False,
    "start_from": "center",
    "stop_after": "test",
    "resume": False,
    "dry_run": False,
    # Center source and output
    "center_json": None,
    "manual_off_detector": None,
    "center_output": None,
    # Center scan parameters. None means inherit the corresponding FBP value.
    "center_projection_glob": None,
    "center_flat_glob": None,
    "center_flat_field": None,
    "center_dark_value": None,
    "center_flat_eps": None,
    "center_processed_npy": None,
    "center_angles_npy": None,
    "center_endpoint_pair_npy": None,
    "center_config": None,
    "center_angle_start": None,
    "center_angle_interval": None,
    "center_keep_duplicate_endpoint": None,
    "center_input_type": None,
    "center_i0": None,
    "center_i0_mode": None,
    "center_free_ray": None,
    "center_i0_percentile": None,
    "center_i0_hi_percentile": None,
    "center_zero_policy": None,
    "center_log_eps": None,
    "center_clip_percentile": None,
    "center_shift_v": None,
    "center_pixel_subsample": 1,
    "center_resize_order": None,
    "center_projection_scale": None,
    "center_pixel_size": None,
    "center_method": "pair",
    "center_init_px": None,
    "center_tol": 0.25,
    "center_algorithm": "scipy",
    "center_search_min_px": -100.0,
    "center_search_max_px": 100.0,
    "center_slice_step": 32,
    "center_slice_margin": 80,
    "center_max_slices": 9,
    "center_pair_min_offset_px": -100.0,
    "center_pair_max_offset_px": 100.0,
    "center_pair_coarse_step_px": 1.0,
    "center_pair_fine_step_px": 0.1,
    "center_pair_min_score": 0.10,
    "center_pair_min_peak_prominence": 0.001,
    "center_pair_max_center_spread_px": 5.0,
    "center_axis_slope_threshold": 0.01,
    "center_threads": 8,
    # FBP/FDK parameters
    "fbp_projection_glob": None,
    "fbp_flat_glob": None,
    "fbp_flat_field": False,
    "fbp_dark_value": 0.0,
    "fbp_flat_eps": 1e-6,
    "fbp_preprocess_only": False,
    "fbp_config": None,
    "fbp_angle_start": 0.0,
    "fbp_angle_interval": 0.5,
    "fbp_keep_duplicate_endpoint": False,
    "fbp_mode": "parallel",
    "fbp_coordinate_mode": "physical",
    "fbp_input_type": "line_integral",
    "fbp_i0": 1.0,
    "fbp_i0_mode": "percentile",
    "fbp_free_ray": [0, 128, 0, 128],
    "fbp_i0_percentile": 99.5,
    "fbp_i0_hi_percentile": 99.9,
    "fbp_zero_policy": "nearest",
    "fbp_log_eps": 1e-6,
    "fbp_clip_percentile": 99.9,
    "fbp_shift_v": 0,
    "fbp_pixel_subsample": 4,
    "fbp_recon_pixel_subsample": None,
    "fbp_resize_order": 1,
    "fbp_projection_scale": 0.125,
    "fbp_n_train": 50,
    "fbp_n_test": 100,
    "fbp_sparse_views": None,
    "fbp_sparse_fbp_output": None,
    "fbp_nVoxel": [128, 128, 128],
    "fbp_sVoxel": [2.0, 2.0, 2.0],
    "fbp_offOrigin": [0.0, 0.0, 0.0],
    "fbp_offDetector": [0.0, 0.0],
    "fbp_DSD": 7.0,
    "fbp_DSO": 5.0,
    "fbp_pixel_size": 0.02,
    "fbp_accuracy": 0.5,
    "fbp_filter": "hann",
    # Initialization
    "init_source": "full_fbp",
    "init_volume": None,
    "init_fbp_output": None,
    "init_output": None,
    "init_n_points": 50000,
    "init_density_thresh": 0.05,
    "init_density_rescale": 0.15,
    "init_normalize_percentile": 99.5,
    "init_seed": None,
    "init_visualize": True,
    "init_visualize_show": False,
    "init_visualize_output": None,
    "init_visualize_max_points": 50000,
    "init_visualize_slices": True,
    "init_visualize_slice_output": None,
    "init_visualize_fbp_nii": False,
    "init_fbp_nii_output": None,
    # Training parameters
    "train_source_path": None,
    "train_model_path": None,
    "train_data_device": "cuda",
    "train_ply_path": None,
    "train_scale_min": 0.0005,
    "train_scale_max": 0.5,
    "train_compute_cov3D_python": False,
    "train_debug": False,
    "train_iterations": 30000,
    "train_position_lr_init": 0.0002,
    "train_position_lr_final": 0.00002,
    "train_position_lr_max_steps": 30000,
    "train_density_lr_init": 0.01,
    "train_density_lr_final": 0.001,
    "train_density_lr_max_steps": 30000,
    "train_scaling_lr_init": 0.005,
    "train_scaling_lr_final": 0.0005,
    "train_scaling_lr_max_steps": 30000,
    "train_rotation_lr_init": 0.001,
    "train_rotation_lr_final": 0.0001,
    "train_rotation_lr_max_steps": 30000,
    "train_lambda_dssim": 0.25,
    "train_lambda_tv": 0.05,
    "train_tv_vol_size": 32,
    "train_density_min_threshold": 0.00001,
    "train_densification_interval": 100,
    "train_densify_from_iter": 500,
    "train_densify_until_iter": 15000,
    "train_densify_grad_threshold": 5.0e-5,
    "train_densify_scale_threshold": 0.1,
    "train_max_screen_size": None,
    "train_max_scale": None,
    "train_max_num_gaussians": 500000,
    "train_slice_loss_enabled": False,
    "train_slice_reference_path": None,
    "train_slice_axis": "z",
    "train_slice_start_iter": 500,
    "train_slice_end_iter": 15000,
    "train_slice_interval": 50,
    "train_slice_filter_threshold": 0.01,
    "train_slice_filter_ratio": 10.0,
    "train_slice_max_points": 200000,
    "train_slice_query_chunk_size": 1024,
    "train_lambda_slice": 0.05,
    "train_slice_seed": 0,
    "train_pose_enabled": False,
    "train_pose_mode": "auto",
    "train_parallel_optimize_angle": False,
    "train_parallel_optimize_global_detector_shift": False,
    "train_parallel_optimize_per_view_detector_shift": False,
    "train_parallel_optimize_beam_tilt": False,
    "train_parallel_optimize_detector_roll": False,
    "train_parallel_angle_max_deg": 0.5,
    "train_parallel_detector_shift_max_pixel": 3.0,
    "train_parallel_beam_tilt_max_deg": 0.5,
    "train_parallel_detector_roll_max_deg": 0.5,
    "train_pose_lr": 0.0001,
    "train_pose_lambda_smooth": 0.1,
    "train_cone_optimize_translation": True,
    "train_cone_optimize_quaternion": True,
    "train_pose_lambda_magnitude": 0.0,
    "train_detect_anomaly": False,
    "train_test_iterations": [5000, 10000, 20000],
    "train_save_iterations": [],
    "train_checkpoint_iterations": [],
    "train_start_checkpoint": None,
    "train_quiet": False,
    # Test parameters
    "test_source_path": None,
    "test_model_path": None,
    "test_data_device": "cuda",
    "test_ply_path": None,
    "test_scale_min": 0.0005,
    "test_scale_max": 0.5,
    "test_compute_cov3D_python": False,
    "test_debug": False,
    "test_iteration": -1,
    "test_skip_render_train": False,
    "test_skip_render_test": False,
    "test_skip_recon": False,
    "test_pose_result": None,
    "test_quiet": False,
}


PATH_KEYS = {
    "input_dir", "output_dir", "model_path", "pipeline_config", "fbp_config",
    "train_config", "center_json", "center_output", "center_config", "center_processed_npy",
    "center_angles_npy", "center_endpoint_pair_npy", "fbp_sparse_fbp_output",
    "init_volume", "init_fbp_output", "init_output", "init_visualize_output",
    "init_visualize_slice_output", "init_fbp_nii_output", "train_source_path",
    "train_slice_reference_path",
    "train_model_path", "train_ply_path", "train_start_checkpoint",
    "test_source_path", "test_model_path", "test_ply_path", "test_pose_result",
}


def _add(parser: argparse.ArgumentParser, *flags: str, dest: str, **kwargs: Any) -> None:
    parser.add_argument(*flags, dest=dest, default=argparse.SUPPRESS, **kwargs)


def _add_bool(parser: argparse.ArgumentParser, name: str, dest: str | None = None) -> None:
    dest = dest or name.lstrip("-").replace("-", "_")
    positive = name if name.startswith("--") else "--" + name
    negative = "--no-" + positive[2:]
    _add(parser, positive, dest=dest, action="store_true")
    _add(parser, negative, dest=dest, action="store_false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS,
    )

    _add(parser, "--input_dir", dest="input_dir", type=Path, help="原始 TIFF 投影目录")
    _add(parser, "--output_dir", dest="output_dir", type=Path, help="FBP 数据集输出目录")
    _add(parser, "--model_path", dest="model_path", type=Path, help="Gaussian 模型输出目录")
    _add(parser, "--pipeline_config", "--pipeline-config", dest="pipeline_config", type=Path,
         help="统一 YAML 配置；命令行参数优先于配置文件")
    _add(parser, "--config", dest="fbp_config", type=Path,
         help="兼容旧接口：角度/投影数量 TXT 配置")
    _add(parser, "--train_config", dest="train_config", type=Path,
         help="兼容旧接口：扁平 train.py YAML 配置，不直接传给 train.py")
    _add(parser, "--seed", dest="seed", type=int)
    for name in ("skip_center", "skip_fbp", "skip_init", "skip_train", "skip_test", "resume", "dry_run"):
        _add_bool(parser, "--" + name, name)
    _add(parser, "--start_from", dest="start_from", choices=STAGES)
    _add(parser, "--stop_after", dest="stop_after", choices=STAGES)

    center_source = parser.add_mutually_exclusive_group()
    center_source.add_argument("--center_json", dest="center_json", type=Path, default=argparse.SUPPRESS,
                               help="已有旋转中心 JSON")
    center_source.add_argument("--center_off_detector", "--offDetector", dest="manual_off_detector",
                               nargs=2, type=float, default=argparse.SUPPRESS, metavar=("U", "V"),
                               help="手动指定 [u, v] 探测器偏移")
    _add(parser, "--center_output", dest="center_output", type=Path)

    # Center scan. Generic names are retained as aliases for the old wrapper.
    center_specs = [
        (("--center_projection_glob",), "center_projection_glob", str),
        (("--center_flat_glob",), "center_flat_glob", str),
        (("--center_flat_field",), "center_flat_field", bool),
        (("--center_dark_value",), "center_dark_value", float),
        (("--center_flat_eps",), "center_flat_eps", float),
        (("--center_processed_npy",), "center_processed_npy", Path),
        (("--center_angles_npy",), "center_angles_npy", Path),
        (("--center_endpoint_pair_npy",), "center_endpoint_pair_npy", Path),
        (("--center_config",), "center_config", Path),
        (("--center_angle_start",), "center_angle_start", float),
        (("--center_angle_interval",), "center_angle_interval", float),
        (("--center_keep_duplicate_endpoint",), "center_keep_duplicate_endpoint", bool),
        (("--center_input_type",), "center_input_type", str),
        (("--center_i0",), "center_i0", float),
        (("--center_i0_mode",), "center_i0_mode", str),
        (("--center_i0_percentile",), "center_i0_percentile", float),
        (("--center_i0_hi_percentile",), "center_i0_hi_percentile", float),
        (("--center_zero_policy",), "center_zero_policy", str),
        (("--center_log_eps",), "center_log_eps", float),
        (("--center_clip_percentile",), "center_clip_percentile", float),
        (("--center_shift_v",), "center_shift_v", int),
        (("--center_pixel_subsample",), "center_pixel_subsample", int),
        (("--center_resize_order",), "center_resize_order", int),
        (("--center_projection_scale",), "center_projection_scale", float),
        (("--center_pixel_size",), "center_pixel_size", float),
        (("--center_method",), "center_method", str),
        (("--center_init_px",), "center_init_px", float),
        (("--center_tol",), "center_tol", float),
        (("--center_algorithm",), "center_algorithm", str),
        (("--center_search_min_px",), "center_search_min_px", float),
        (("--center_search_max_px",), "center_search_max_px", float),
        (("--center_slice_step", "--slice_step"), "center_slice_step", int),
        (("--center_slice_margin", "--slice_margin"), "center_slice_margin", int),
        (("--center_max_slices", "--max_slices"), "center_max_slices", int),
        (("--center_pair_min_offset_px",), "center_pair_min_offset_px", float),
        (("--center_pair_max_offset_px",), "center_pair_max_offset_px", float),
        (("--center_pair_coarse_step_px",), "center_pair_coarse_step_px", float),
        (("--center_pair_fine_step_px",), "center_pair_fine_step_px", float),
        (("--center_pair_min_score",), "center_pair_min_score", float),
        (("--center_pair_min_peak_prominence",), "center_pair_min_peak_prominence", float),
        (("--center_pair_max_center_spread_px",), "center_pair_max_center_spread_px", float),
        (("--center_axis_slope_threshold",), "center_axis_slope_threshold", float),
        (("--center_threads",), "center_threads", int),
    ]
    for flags, dest, typ in center_specs:
        if typ is bool:
            _add_bool(parser, flags[0], dest)
        else:
            kwargs: dict[str, Any] = {"type": typ}
            if dest == "center_input_type": kwargs["choices"] = ["transmission", "intensity", "line_integral"]
            if dest == "center_i0_mode": kwargs["choices"] = ["percentile", "free_ray", "fixed"]
            if dest == "center_zero_policy": kwargs["choices"] = ["nearest", "clip", "keep"]
            if dest == "center_method": kwargs["choices"] = ["pair", "vo", "scipy"]
            _add(parser, *flags, dest=dest, **kwargs)
    _add(parser, "--center_free_ray", dest="center_free_ray", nargs=4, type=int,
         metavar=("ROW0", "ROW1", "COL0", "COL1"))

    # FBP/FDK. The old unprefixed options are aliases for the FBP values.
    def fbp_flag(name: str, *aliases: str, **kwargs: Any) -> None:
        _add(parser, "--fbp_" + name, *aliases, dest="fbp_" + name, **kwargs)

    fbp_flag("projection_glob", "--projection_glob", type=str)
    fbp_flag("flat_glob", "--flat_glob", type=str)
    _add_bool(parser, "--fbp_flat_field", "fbp_flat_field")
    _add(parser, "--flat_field", dest="fbp_flat_field", action="store_true")
    fbp_flag("dark_value", "--dark_value", type=float)
    fbp_flag("flat_eps", "--flat_eps", type=float)
    _add_bool(parser, "--fbp_preprocess_only", "fbp_preprocess_only")
    _add(parser, "--preprocess_only", dest="fbp_preprocess_only", action="store_true")
    fbp_flag("config", type=Path)
    fbp_flag("angle_start", "--angle_start", type=float)
    fbp_flag("angle_interval", "--angle_interval", type=float)
    _add_bool(parser, "--fbp_keep_duplicate_endpoint", "fbp_keep_duplicate_endpoint")
    _add(parser, "--keep_duplicate_endpoint", dest="fbp_keep_duplicate_endpoint", action="store_true")
    fbp_flag("mode", "--mode", choices=["parallel", "cone"])
    fbp_flag("coordinate_mode", choices=["physical", "gaussian"])
    fbp_flag("input_type", "--input_type", choices=["transmission", "intensity", "line_integral"])
    fbp_flag("i0", "--i0", type=float)
    fbp_flag("i0_mode", "--i0_mode", choices=["percentile", "free_ray", "fixed"])
    fbp_flag("i0_percentile", "--i0_percentile", type=float)
    fbp_flag("i0_hi_percentile", "--i0_hi_percentile", type=float)
    fbp_flag("zero_policy", "--zero_policy", type=str, choices=["nearest", "clip", "keep"])
    fbp_flag("log_eps", "--log_eps", type=float)
    fbp_flag("clip_percentile", "--clip_percentile", type=float)
    fbp_flag("shift_v", "--shift_v", type=int)
    fbp_flag("pixel_subsample", "--pixel_subsample", type=int)
    fbp_flag("recon_pixel_subsample", "--recon_pixel_subsample", type=int)
    fbp_flag("resize_order", "--resize_order", type=int, choices=[0, 1, 3])
    fbp_flag("projection_scale", "--projection_scale", type=float)
    fbp_flag("n_train", "--n_train", type=int)
    fbp_flag("n_test", "--n_test", type=int)
    fbp_flag("sparse_views", type=int)
    fbp_flag("sparse_fbp_output", type=Path)
    fbp_flag("nVoxel", "--nVoxel", nargs=3, type=int, metavar=("X", "Y", "Z"))
    fbp_flag("sVoxel", "--sVoxel", nargs=3, type=float, metavar=("X", "Y", "Z"))
    fbp_flag("offOrigin", nargs=3, type=float, metavar=("X", "Y", "Z"))
    fbp_flag("offDetector", nargs=2, type=float, metavar=("U", "V"))
    fbp_flag("DSD", "--DSD", type=float)
    fbp_flag("DSO", "--DSO", type=float)
    fbp_flag("pixel_size", "--pixel_size", type=float)
    fbp_flag("accuracy", "--accuracy", type=float)
    fbp_flag("filter", "--filter", choices=["ram_lak", "shepp_logan", "cosine", "hamming", "hann"])
    _add(parser, "--fbp_free_ray", "--free_ray", dest="fbp_free_ray", nargs=4, type=int,
         metavar=("ROW0", "ROW1", "COL0", "COL1"))

    # Initialization.
    _add(parser, "--init_source", dest="init_source", choices=["full_fbp", "train_fbp", "custom"])
    _add(parser, "--init_volume", dest="init_volume", type=Path)
    _add(parser, "--init_fbp_output", dest="init_fbp_output", type=Path)
    _add(parser, "--init_output", dest="init_output", type=Path)
    _add(parser, "--n_points", "--init_n_points", dest="init_n_points", type=int)
    _add(parser, "--density_thresh", "--init_density_thresh", dest="init_density_thresh", type=float)
    _add(parser, "--density_rescale", "--init_density_rescale", dest="init_density_rescale", type=float)
    _add(parser, "--normalize_percentile", "--init_normalize_percentile", dest="init_normalize_percentile", type=float)
    _add(parser, "--init_seed", dest="init_seed", type=int)
    _add_bool(parser, "--init_visualize", "init_visualize")
    _add_bool(parser, "--init_visualize_show", "init_visualize_show")
    _add(parser, "--init_visualize_output", dest="init_visualize_output", type=Path)
    _add(parser, "--init_visualize_max_points", dest="init_visualize_max_points", type=int)
    _add_bool(parser, "--init_visualize_slices", "init_visualize_slices")
    _add(parser, "--init_visualize_slice_output", dest="init_visualize_slice_output", type=Path)
    _add_bool(parser, "--init_visualize_fbp_nii", "init_visualize_fbp_nii")
    _add(parser, "--init_fbp_nii_output", dest="init_fbp_nii_output", type=Path)

    # Training parameters. These map one-to-one to train.py's ParamGroups.
    train_specs: list[tuple[str, Any]] = [
        ("source_path", Path), ("model_path", Path), ("data_device", str), ("ply_path", Path),
        ("scale_min", float), ("scale_max", float), ("compute_cov3D_python", bool), ("debug", bool),
        ("iterations", int), ("position_lr_init", float), ("position_lr_final", float),
        ("position_lr_max_steps", int), ("density_lr_init", float), ("density_lr_final", float),
        ("density_lr_max_steps", int), ("scaling_lr_init", float), ("scaling_lr_final", float),
        ("scaling_lr_max_steps", int), ("rotation_lr_init", float), ("rotation_lr_final", float),
        ("rotation_lr_max_steps", int), ("lambda_dssim", float), ("lambda_tv", float),
        ("tv_vol_size", int), ("density_min_threshold", float), ("densification_interval", int),
        ("densify_from_iter", int), ("densify_until_iter", int), ("densify_grad_threshold", float),
        ("densify_scale_threshold", float), ("max_screen_size", float), ("max_scale", float),
        ("max_num_gaussians", int), ("slice_loss_enabled", bool),
        ("slice_reference_path", Path), ("slice_axis", str),
        ("slice_start_iter", int), ("slice_end_iter", int),
        ("slice_interval", int), ("slice_filter_threshold", float),
        ("slice_filter_ratio", float), ("slice_max_points", int),
        ("slice_query_chunk_size", int), ("lambda_slice", float),
        ("slice_seed", int), ("pose_enabled", bool), ("pose_mode", str),
        ("parallel_optimize_angle", bool), ("parallel_optimize_global_detector_shift", bool),
        ("parallel_optimize_per_view_detector_shift", bool), ("parallel_optimize_beam_tilt", bool),
        ("parallel_optimize_detector_roll", bool), ("parallel_angle_max_deg", float),
        ("parallel_detector_shift_max_pixel", float), ("parallel_beam_tilt_max_deg", float),
        ("parallel_detector_roll_max_deg", float), ("pose_lr", float), ("pose_lambda_smooth", float),
        ("cone_optimize_translation", bool), ("cone_optimize_quaternion", bool),
        ("pose_lambda_magnitude", float),
    ]
    for name, typ in train_specs:
        dest = "train_" + name
        if typ is bool:
            _add_bool(parser, "--train_" + name, dest)
        else:
            _add(parser, "--train_" + name, dest=dest, type=typ)
    _add(parser, "--train_test_iterations", dest="train_test_iterations", nargs="+", type=int)
    _add(parser, "--train_save_iterations", dest="train_save_iterations", nargs="+", type=int)
    _add(parser, "--train_checkpoint_iterations", dest="train_checkpoint_iterations", nargs="+", type=int)
    _add(parser, "--train_start_checkpoint", dest="train_start_checkpoint", type=Path)
    _add_bool(parser, "--train_detect_anomaly", "train_detect_anomaly")
    _add_bool(parser, "--train_quiet", "train_quiet")
    _add(parser, "--iterations", dest="train_iterations", type=int)

    # Test parameters.
    test_specs: list[tuple[str, Any]] = [
        ("source_path", Path), ("model_path", Path), ("data_device", str), ("ply_path", Path),
        ("scale_min", float), ("scale_max", float), ("compute_cov3D_python", bool), ("debug", bool),
    ]
    for name, typ in test_specs:
        dest = "test_" + name
        if typ is bool:
            _add_bool(parser, "--test_" + name, dest)
        else:
            _add(parser, "--test_" + name, dest=dest, type=typ)
    _add(parser, "--test_iteration", "--iteration", dest="test_iteration", type=int)
    _add_bool(parser, "--test_skip_render_train", "test_skip_render_train")
    _add_bool(parser, "--test_skip_render_test", "test_skip_render_test")
    _add_bool(parser, "--test_skip_recon", "test_skip_recon")
    _add(parser, "--test_pose_result", "--pose_result", dest="test_pose_result", type=Path)
    _add_bool(parser, "--test_quiet", "test_quiet")
    _add_bool(parser, "--skip_render_train", "test_skip_render_train")
    _add_bool(parser, "--skip_render_test", "test_skip_render_test")
    _add_bool(parser, "--skip_recon", "test_skip_recon")

    return parser


SECTION_PREFIX = {"center": "center_", "fbp": "fbp_", "init": "init_", "train": "train_", "test": "test_"}


def _config_key(section: str | None, key: str) -> str:
    key = str(key).replace("-", "_")
    if section in SECTION_PREFIX:
        prefix = SECTION_PREFIX[section]
        return key if key.startswith(prefix) else prefix + key
    if section in {"paths", "global", "pipeline"}:
        return key
    if section is None and ("train_" + key) in DEFAULTS:
        return "train_" + key
    aliases = {
        "config": "fbp_config", "mode": "fbp_mode", "input_type": "fbp_input_type",
        "i0": "fbp_i0", "i0_mode": "fbp_i0_mode", "free_ray": "fbp_free_ray",
        "pixel_subsample": "fbp_pixel_subsample", "projection_scale": "fbp_projection_scale",
        "pixel_size": "fbp_pixel_size", "iterations": "train_iterations",
        "n_points": "init_n_points", "density_thresh": "init_density_thresh",
        "density_rescale": "init_density_rescale", "normalize_percentile": "init_normalize_percentile",
        "skip_render_train": "test_skip_render_train", "skip_render_test": "test_skip_render_test",
        "skip_recon": "test_skip_recon", "pose_result": "test_pose_result",
        "offDetector": "manual_off_detector",
    }
    return aliases.get(key, key)


def flatten_config(data: dict[str, Any], legacy_train: bool = False) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Pipeline YAML must contain a mapping at its top level")
    result: dict[str, Any] = {}
    for section, values in data.items():
        if legacy_train and section == "inherit_from":
            continue
        if isinstance(values, dict):
            section_name = "train" if legacy_train else str(section)
            for key, value in values.items():
                result[_config_key(section_name, str(key))] = value
        else:
            section_name = None
            result[_config_key(section_name, str(section))] = values
    return result


def load_yaml(path: Path, legacy_train: bool = False) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return flatten_config(data, legacy_train=legacy_train)


def resolve_args(parsed: argparse.Namespace) -> argparse.Namespace:
    values = dict(DEFAULTS)

    def merge_loaded(loaded: dict[str, Any], source: Path) -> None:
        unknown = sorted(set(loaded) - set(DEFAULTS))
        if unknown:
            raise ValueError(f"Unknown keys in {source}: {', '.join(unknown)}")
        values.update(loaded)

    pipeline_config = getattr(parsed, "pipeline_config", None)
    train_config = getattr(parsed, "train_config", None)
    if train_config is not None:
        train_config = Path(train_config).resolve()
        merge_loaded(load_yaml(train_config, legacy_train=True), train_config)
    if pipeline_config is not None:
        pipeline_config = Path(pipeline_config).resolve()
        merge_loaded(load_yaml(pipeline_config), pipeline_config)
    values.update(vars(parsed))
    for key in PATH_KEYS:
        if values.get(key) is not None and not isinstance(values[key], Path):
            values[key] = Path(values[key])
    for key in ("fbp_nVoxel", "fbp_sVoxel", "fbp_offOrigin", "fbp_offDetector", "fbp_free_ray", "manual_off_detector"):
        if values.get(key) is not None:
            values[key] = list(values[key])

    inherit = {
        "center_projection_glob": "fbp_projection_glob", "center_flat_glob": "fbp_flat_glob",
        "center_flat_field": "fbp_flat_field", "center_dark_value": "fbp_dark_value",
        "center_flat_eps": "fbp_flat_eps", "center_config": "fbp_config",
        "center_angle_start": "fbp_angle_start", "center_angle_interval": "fbp_angle_interval",
        "center_keep_duplicate_endpoint": "fbp_keep_duplicate_endpoint", "center_input_type": "fbp_input_type",
        "center_i0": "fbp_i0", "center_i0_mode": "fbp_i0_mode", "center_free_ray": "fbp_free_ray",
        "center_i0_percentile": "fbp_i0_percentile", "center_i0_hi_percentile": "fbp_i0_hi_percentile",
        "center_zero_policy": "fbp_zero_policy", "center_log_eps": "fbp_log_eps",
        "center_clip_percentile": "fbp_clip_percentile", "center_shift_v": "fbp_shift_v",
        "center_resize_order": "fbp_resize_order", "center_projection_scale": "fbp_projection_scale",
        "center_pixel_size": "fbp_pixel_size",
    }
    for key, source in inherit.items():
        if values[key] is None:
            values[key] = values[source]
    # Preserve the old wrapper's center_search_min/max -> pair scan behavior
    # when the newer pair-specific options were not supplied.
    if values["center_pair_min_offset_px"] == DEFAULTS["center_pair_min_offset_px"] and values["center_search_min_px"] != DEFAULTS["center_search_min_px"]:
        values["center_pair_min_offset_px"] = values["center_search_min_px"]
    if values["center_pair_max_offset_px"] == DEFAULTS["center_pair_max_offset_px"] and values["center_search_max_px"] != DEFAULTS["center_search_max_px"]:
        values["center_pair_max_offset_px"] = values["center_search_max_px"]
    if values["center_output"] is None and values["output_dir"] is not None:
        output = Path(values["output_dir"])
        values["center_output"] = output.with_name(output.name + "_center.json")
    values["init_seed"] = values["seed"] if values["init_seed"] is None else values["init_seed"]
    if values.get("train_slice_loss_enabled") and values.get("train_slice_reference_path") is None:
        if values.get("init_source") == "custom":
            values["train_slice_reference_path"] = values.get("init_volume")
        elif values.get("init_source") == "train_fbp":
            output = Path(values["output_dir"])
            values["train_slice_reference_path"] = (
                values.get("init_fbp_output") or output / (
                    "vol_train_fbp_gaussian.npy" if values["fbp_coordinate_mode"] == "gaussian"
                    else "vol_train_fbp.npy"
                )
            )
        else:
            values["train_slice_reference_path"] = Path(values["output_dir"]) / "vol_fbp.npy"
    return argparse.Namespace(**values)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(manifest), handle, indent=2, ensure_ascii=False)


def add_option(command: list[str], flag: str, value: Any, *, bool_flag: bool = False) -> None:
    if value is None:
        return
    if bool_flag:
        if bool(value):
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        command.append(flag)
        command.extend(str(item) for item in value)
    else:
        command.extend([flag, str(value)])


def add_bool(command: list[str], flag: str, value: Any) -> None:
    add_option(command, flag, value, bool_flag=True)


def build_center_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / "scan_fbp_center_tomopy.py")]
    if args.center_processed_npy is not None:
        add_option(command, "--processed_npy", args.center_processed_npy)
        add_option(command, "--angles_npy", args.center_angles_npy)
        add_option(command, "--endpoint_pair_npy", args.center_endpoint_pair_npy)
    else:
        add_option(command, "--input_dir", args.input_dir)
        add_option(command, "--projection_glob", args.center_projection_glob)
    add_option(command, "--flat_glob", args.center_flat_glob)
    add_bool(command, "--flat_field", args.center_flat_field)
    for name, flag in (
        ("center_dark_value", "--dark_value"), ("center_flat_eps", "--flat_eps"),
        ("center_config", "--config"), ("center_angle_start", "--angle_start"),
        ("center_angle_interval", "--angle_interval"), ("center_input_type", "--input_type"),
        ("center_i0", "--i0"), ("center_i0_mode", "--i0_mode"),
        ("center_i0_percentile", "--i0_percentile"), ("center_i0_hi_percentile", "--i0_hi_percentile"),
        ("center_zero_policy", "--zero_policy"), ("center_log_eps", "--log_eps"),
        ("center_clip_percentile", "--clip_percentile"), ("center_shift_v", "--shift_v"),
        ("center_pixel_subsample", "--pixel_subsample"), ("center_resize_order", "--resize_order"),
        ("center_projection_scale", "--projection_scale"), ("center_pixel_size", "--pixel_size"),
        ("center_method", "--method"), ("center_init_px", "--init_px"), ("center_tol", "--tol"),
        ("center_algorithm", "--algorithm"), ("center_search_min_px", "--search_min_px"),
        ("center_search_max_px", "--search_max_px"), ("center_slice_step", "--slice_step"),
        ("center_slice_margin", "--slice_margin"), ("center_max_slices", "--max_slices"),
        ("center_pair_min_offset_px", "--pair_min_offset_px"), ("center_pair_max_offset_px", "--pair_max_offset_px"),
        ("center_pair_coarse_step_px", "--pair_coarse_step_px"), ("center_pair_fine_step_px", "--pair_fine_step_px"),
        ("center_pair_min_score", "--pair_min_score"),
        ("center_pair_min_peak_prominence", "--pair_min_peak_prominence"),
        ("center_pair_max_center_spread_px", "--pair_max_center_spread_px"),
        ("center_axis_slope_threshold", "--axis_slope_threshold"), ("center_threads", "--threads"),
    ):
        add_option(command, flag, getattr(args, name))
    add_option(command, "--free_ray", args.center_free_ray)
    add_bool(command, "--keep_duplicate_endpoint", args.center_keep_duplicate_endpoint)
    add_option(command, "--output", args.center_output)
    return command


def build_fbp_command(args: argparse.Namespace, center_json: Path | None, manual_offset: list[float] | None) -> list[str]:
    command = [sys.executable, str(ROOT / "prepare_fbp_tiff.py"), "--input_dir", str(args.input_dir), "--output_dir", str(args.output_dir)]
    mapping = (
        ("fbp_projection_glob", "--projection_glob"), ("fbp_flat_glob", "--flat_glob"),
        ("fbp_dark_value", "--dark_value"), ("fbp_flat_eps", "--flat_eps"),
        ("fbp_config", "--config"), ("fbp_angle_start", "--angle_start"),
        ("fbp_angle_interval", "--angle_interval"), ("fbp_mode", "--mode"),
        ("fbp_coordinate_mode", "--coordinate_mode"),
        ("fbp_input_type", "--input_type"), ("fbp_i0", "--i0"),
        ("fbp_i0_mode", "--i0_mode"), ("fbp_i0_percentile", "--i0_percentile"),
        ("fbp_i0_hi_percentile", "--i0_hi_percentile"), ("fbp_zero_policy", "--zero_policy"),
        ("fbp_log_eps", "--log_eps"), ("fbp_clip_percentile", "--clip_percentile"),
        ("fbp_shift_v", "--shift_v"), ("fbp_pixel_subsample", "--pixel_subsample"),
        ("fbp_recon_pixel_subsample", "--recon_pixel_subsample"), ("fbp_resize_order", "--resize_order"),
        ("fbp_projection_scale", "--projection_scale"), ("fbp_n_train", "--n_train"),
        ("fbp_n_test", "--n_test"), ("fbp_sparse_views", "--sparse_views"),
        ("fbp_sparse_fbp_output", "--sparse_fbp_output"), ("fbp_nVoxel", "--nVoxel"),
        ("fbp_sVoxel", "--sVoxel"), ("fbp_offOrigin", "--offOrigin"),
        ("fbp_DSD", "--DSD"), ("fbp_DSO", "--DSO"), ("fbp_pixel_size", "--pixel_size"),
        ("fbp_accuracy", "--accuracy"), ("fbp_filter", "--filter"),
    )
    for name, flag in mapping:
        add_option(command, flag, getattr(args, name))
    add_option(command, "--seed", args.seed)
    add_option(command, "--free_ray", args.fbp_free_ray)
    add_bool(command, "--flat_field", args.fbp_flat_field)
    add_bool(command, "--preprocess_only", args.fbp_preprocess_only)
    add_bool(command, "--keep_duplicate_endpoint", args.fbp_keep_duplicate_endpoint)
    if center_json is not None:
        add_option(command, "--center_json", center_json)
    else:
        add_option(command, "--offDetector", manual_offset if manual_offset is not None else args.fbp_offDetector)
    return command


def build_init_command(args: argparse.Namespace, init_output: Path) -> list[str]:
    command = [sys.executable, str(ROOT / "init_from_fbp.py")]
    if args.init_source == "full_fbp":
        add_option(command, "--volume", args.output_dir / "vol_fbp.npy")
    elif args.init_source == "train_fbp":
        add_option(command, "--dataset", args.output_dir)
        add_bool(command, "--use_train_fbp", True)
        add_option(command, "--coordinate_mode", args.fbp_coordinate_mode)
        add_option(command, "--fbp_output", args.init_fbp_output)
    else:
        if args.init_volume is None:
            raise ValueError("--init_volume is required when --init_source=custom")
        add_option(command, "--volume", args.init_volume)
    add_option(command, "--output", init_output)
    for name, flag in (
        ("init_n_points", "--n_points"), ("init_density_thresh", "--density_thresh"),
        ("init_density_rescale", "--density_rescale"), ("init_normalize_percentile", "--normalize_percentile"),
        ("init_seed", "--seed"), ("init_visualize_output", "--visualize_output"),
        ("init_visualize_max_points", "--visualize_max_points"),
        ("init_visualize_slice_output", "--visualize_slice_output"),
        ("init_fbp_nii_output", "--fbp_nii_output"),
    ):
        add_option(command, flag, getattr(args, name))
    add_bool(command, "--visualize", args.init_visualize)
    add_bool(command, "--visualize_show", args.init_visualize_show)
    add_bool(command, "--visualize_slices", args.init_visualize_slices)
    add_bool(command, "--visualize_fbp_nii", args.init_visualize_fbp_nii)
    return command


def write_train_config(args: argparse.Namespace, path: Path, source: Path, model: Path, ply: Path) -> None:
    """Write a flat train.py config so options unsupported by argparse's
    legacy None-typed arguments (for example max_scale) still work."""
    names = (
        "data_device", "ply_path", "scale_min", "scale_max", "iterations",
        "position_lr_init", "position_lr_final", "position_lr_max_steps", "density_lr_init",
        "density_lr_final", "density_lr_max_steps", "scaling_lr_init", "scaling_lr_final",
        "scaling_lr_max_steps", "rotation_lr_init", "rotation_lr_final", "rotation_lr_max_steps",
        "lambda_dssim", "lambda_tv", "tv_vol_size", "density_min_threshold",
        "densification_interval", "densify_from_iter", "densify_until_iter", "densify_grad_threshold",
        "densify_scale_threshold", "max_screen_size", "max_scale", "max_num_gaussians",
        "slice_loss_enabled", "slice_reference_path", "slice_axis",
        "slice_start_iter", "slice_end_iter", "slice_interval",
        "slice_filter_threshold", "slice_filter_ratio", "slice_max_points",
        "slice_query_chunk_size", "lambda_slice", "slice_seed",
        "pose_enabled", "pose_mode", "parallel_optimize_angle", "parallel_optimize_global_detector_shift",
        "parallel_optimize_per_view_detector_shift", "parallel_optimize_beam_tilt",
        "parallel_optimize_detector_roll", "parallel_angle_max_deg", "parallel_detector_shift_max_pixel",
        "parallel_beam_tilt_max_deg", "parallel_detector_roll_max_deg", "pose_lr", "pose_lambda_smooth",
        "cone_optimize_translation", "cone_optimize_quaternion", "pose_lambda_magnitude",
        "detect_anomaly", "test_iterations", "save_iterations", "checkpoint_iterations",
        "start_checkpoint", "quiet",
    )
    values: dict[str, Any] = {"source_path": str(source), "model_path": str(model), "ply_path": str(ply)}
    for name in names:
        if name == "test_iterations":
            value = args.train_test_iterations
        elif name == "save_iterations":
            value = args.train_save_iterations
        elif name == "checkpoint_iterations":
            value = args.train_checkpoint_iterations
        elif name == "start_checkpoint":
            value = args.train_start_checkpoint
        elif name == "detect_anomaly":
            value = args.train_detect_anomaly
        elif name == "quiet":
            value = args.train_quiet
        else:
            value = getattr(args, "train_" + name)
        if value is not None:
            values[name] = _jsonable(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, sort_keys=False, allow_unicode=True)


def build_train_command(args: argparse.Namespace, init_output: Path, train_config_path: Path) -> list[str]:
    source = args.train_source_path or args.output_dir
    model = args.train_model_path or args.model_path
    ply = args.train_ply_path or init_output
    command = [sys.executable, str(ROOT / "train.py"), "-s", str(source), "-m", str(model)]
    add_option(command, "--ply_path", ply)
    add_option(command, "--config", train_config_path)
    scalar_names = (
        "data_device", "scale_min", "scale_max", "iterations", "position_lr_init", "position_lr_final",
        "position_lr_max_steps", "density_lr_init", "density_lr_final", "density_lr_max_steps",
        "scaling_lr_init", "scaling_lr_final", "scaling_lr_max_steps", "rotation_lr_init",
        "rotation_lr_final", "rotation_lr_max_steps", "lambda_dssim", "lambda_tv", "tv_vol_size",
        "density_min_threshold", "densification_interval", "densify_from_iter", "densify_until_iter",
        "densify_grad_threshold", "densify_scale_threshold", "max_screen_size", "max_scale",
        "max_num_gaussians", "slice_reference_path", "slice_axis", "slice_start_iter",
        "slice_end_iter", "slice_interval", "slice_filter_threshold",
        "slice_filter_ratio", "slice_max_points", "slice_query_chunk_size",
        "lambda_slice", "slice_seed", "pose_mode", "parallel_angle_max_deg", "parallel_detector_shift_max_pixel",
        "parallel_beam_tilt_max_deg", "parallel_detector_roll_max_deg", "pose_lr", "pose_lambda_smooth",
        "pose_lambda_magnitude",
    )
    for name in scalar_names:
        add_option(command, "--" + name, getattr(args, "train_" + name))
    for name in ("compute_cov3D_python", "debug", "slice_loss_enabled", "pose_enabled", "parallel_optimize_angle",
                 "parallel_optimize_global_detector_shift", "parallel_optimize_per_view_detector_shift",
                 "parallel_optimize_beam_tilt", "parallel_optimize_detector_roll",
                 "cone_optimize_translation", "cone_optimize_quaternion", "detect_anomaly", "quiet"):
        add_bool(command, "--" + name, getattr(args, "train_" + name))
    for name, flag in (("train_test_iterations", "--test_iterations"), ("train_save_iterations", "--save_iterations"),
                       ("train_checkpoint_iterations", "--checkpoint_iterations")):
        values = getattr(args, name)
        if values:
            add_option(command, flag, values)
    add_option(command, "--start_checkpoint", args.train_start_checkpoint)
    return command


def build_test_command(args: argparse.Namespace, pose_result: Path | None) -> list[str]:
    source = args.test_source_path or args.output_dir
    model = args.test_model_path or args.model_path
    command = [sys.executable, str(ROOT / "test.py"), "-s", str(source), "-m", str(model)]
    add_option(command, "--iteration", args.test_iteration)
    for name in ("data_device", "ply_path", "scale_min", "scale_max"):
        add_option(command, "--" + name, getattr(args, "test_" + name))
    add_bool(command, "--compute_cov3D_python", args.test_compute_cov3D_python)
    add_bool(command, "--debug", args.test_debug)
    add_bool(command, "--skip_render_train", args.test_skip_render_train)
    add_bool(command, "--skip_render_test", args.test_skip_render_test)
    add_bool(command, "--skip_recon", args.test_skip_recon)
    add_bool(command, "--quiet", args.test_quiet)
    add_option(command, "--pose_result", pose_result)
    return command


def validate_args(args: argparse.Namespace) -> None:
    missing = [name for name in ("input_dir", "output_dir", "model_path") if getattr(args, name) is None]
    if missing:
        raise ValueError("Missing required arguments: " + ", ".join("--" + name for name in missing))
    if args.center_json is not None and args.manual_off_detector is not None:
        raise ValueError("--center_json and --center_off_detector/--offDetector are mutually exclusive")
    if args.skip_center and (args.center_json is not None or args.manual_off_detector is not None):
        raise ValueError("--skip_center cannot be combined with an explicit center source")
    if args.center_pixel_subsample < 1 or args.fbp_pixel_subsample < 1:
        raise ValueError("pixel_subsample values must be at least 1")
    if args.fbp_recon_pixel_subsample is not None and args.fbp_recon_pixel_subsample < 1:
        raise ValueError("fbp_recon_pixel_subsample must be at least 1")
    if args.fbp_mode == "cone" and args.fbp_DSD <= args.fbp_DSO:
        raise ValueError("fbp_mode=cone requires fbp_DSD > fbp_DSO")
    if args.fbp_coordinate_mode == "gaussian" and args.fbp_mode != "parallel":
        raise ValueError("fbp_coordinate_mode=gaussian requires fbp_mode=parallel")
    if args.center_search_min_px >= args.center_search_max_px:
        raise ValueError("center_search_min_px must be smaller than center_search_max_px")
    if args.center_pair_min_offset_px >= args.center_pair_max_offset_px:
        raise ValueError("center_pair_min_offset_px must be smaller than center_pair_max_offset_px")
    if any(int(value) <= 0 for value in args.fbp_nVoxel) or any(float(value) <= 0 for value in args.fbp_sVoxel):
        raise ValueError("fbp_nVoxel and fbp_sVoxel must contain positive values")
    if args.fbp_n_train < 1 or args.fbp_n_test < 0:
        raise ValueError("fbp_n_train must be positive and fbp_n_test cannot be negative")
    if args.init_n_points < 1 or args.init_density_thresh < 0:
        raise ValueError("init_n_points must be positive and init_density_thresh cannot be negative")
    if args.init_fbp_nii_output is not None and not args.init_visualize_fbp_nii:
        raise ValueError("init_fbp_nii_output requires init_visualize_fbp_nii")
    if args.init_visualize_fbp_nii and args.init_source != "train_fbp":
        raise ValueError("init_visualize_fbp_nii requires init_source=train_fbp")
    if args.train_pose_mode not in {"auto", "parallel", "cone"}:
        raise ValueError("train_pose_mode must be auto, parallel, or cone")
    if args.train_pose_enabled and args.train_pose_mode != "auto" and args.train_pose_mode != args.fbp_mode:
        raise ValueError("train_pose_mode must match fbp_mode when pose calibration is enabled")
    if args.train_slice_loss_enabled:
        if args.train_pose_enabled:
            raise ValueError("train_slice_loss_enabled cannot be combined with train_pose_enabled")
        if args.train_slice_axis not in {"x", "y", "z"}:
            raise ValueError("train_slice_axis must be x, y, or z")
        if args.train_slice_interval < 1 or args.train_slice_max_points < 1:
            raise ValueError("slice interval and max points must be positive")
        if args.train_slice_end_iter < args.train_slice_start_iter:
            raise ValueError("train_slice_end_iter must be >= train_slice_start_iter")
        if args.train_slice_query_chunk_size < 1:
            raise ValueError("train_slice_query_chunk_size must be positive")
        if not (0.0 < args.train_slice_filter_ratio <= 100.0):
            raise ValueError("train_slice_filter_ratio must be in (0, 100]")
        if args.train_lambda_slice < 0:
            raise ValueError("train_lambda_slice must be non-negative")
        if args.train_slice_reference_path is None:
            raise ValueError("slice reference path is required when slice loss is enabled")


def stage_in_window(args: argparse.Namespace, stage: str) -> bool:
    lo = STAGES.index(args.start_from)
    hi = STAGES.index(args.stop_after)
    index = STAGES.index(stage)
    if lo > hi:
        raise ValueError("start_from must not come after stop_after")
    if not (lo <= index <= hi):
        return False
    return not bool(getattr(args, "skip_" + stage))


def artifact_for_init(args: argparse.Namespace, init_output: Path) -> Path:
    return Path(args.train_ply_path or init_output).resolve()


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    args.input_dir = Path(args.input_dir).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.model_path = Path(args.model_path).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fbp_preprocess_only and stage_in_window(args, "init"):
        raise ValueError("fbp_preprocess_only requires stop_after=fbp or skipping init/train/test")
    manifest_path = args.output_dir / "pipeline_manifest.json"
    init_output = (args.init_output or args.output_dir / ("init_" + args.output_dir.name + ".npy")).resolve()
    center_output = Path(args.center_output).resolve()
    args.center_output = center_output
    args.init_output = init_output

    if args.init_source == "custom" and args.init_volume is not None:
        args.init_volume = Path(args.init_volume).resolve()
    if args.center_json is not None:
        args.center_json = Path(args.center_json).resolve()

    manifest: dict[str, Any] = {
        "version": 2,
        "root": str(ROOT),
        "resolved_arguments": vars(args),
        "stage_commands": {},
        "stage_status": {stage: "pending" for stage in STAGES},
        "center_output": center_output,
        "init_output": init_output,
    }
    previous: dict[str, Any] = {}
    if args.resume and manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    def save() -> None:
        manifest["resolved_arguments"] = vars(args)
        write_manifest(manifest_path, manifest)

    def run_stage(stage: str, stage_command: list[str]) -> None:
        manifest["stage_commands"][stage] = stage_command
        if args.resume and previous.get("stage_status", {}).get(stage) == "success":
            manifest["stage_status"][stage] = "resumed"
            print(f"[{stage}] resume: previous success retained", flush=True)
            if stage not in {"center", "fbp"}:
                save()
            return
        if args.dry_run:
            manifest["stage_status"][stage] = "planned"
            print("[dry-run]", " ".join(str(item) for item in stage_command), flush=True)
            save()
            return
        manifest["stage_status"][stage] = "running"
        # prepare_fbp_tiff.py requires a fresh output directory. Do not write
        # the manifest into it before the FBP stage has started/completed.
        if stage not in {"center", "fbp"}:
            save()
        print("+", " ".join(str(item) for item in stage_command), flush=True)
        try:
            subprocess.run(stage_command, cwd=ROOT, check=True)
        except Exception:
            manifest["stage_status"][stage] = "failed"
            if stage not in {"center", "fbp"}:
                save()
            raise
        manifest["stage_status"][stage] = "success"
        if stage != "center":
            save()

    center_stage = stage_in_window(args, "center") and args.center_json is None and args.manual_off_detector is None and not args.skip_center
    if center_stage:
        run_stage("center", build_center_command(args))
        center_for_fbp = center_output
        manual_for_fbp = None
    else:
        if args.center_json is not None:
            center_for_fbp = args.center_json
            manual_for_fbp = None
        elif args.manual_off_detector is not None:
            center_for_fbp = None
            manual_for_fbp = args.manual_off_detector
        else:
            center_for_fbp = None
            manual_for_fbp = args.fbp_offDetector
            if not stage_in_window(args, "center"):
                print("Center stage skipped; using fbp_offDetector.", flush=True)
    manifest["stage_status"]["center"] = manifest["stage_status"]["center"] if center_stage else "skipped"

    fbp_stage = stage_in_window(args, "fbp")
    if fbp_stage:
        run_stage("fbp", build_fbp_command(args, center_for_fbp, manual_for_fbp))
    else:
        manifest["stage_status"]["fbp"] = "skipped"
        if not args.dry_run and args.init_source in {"full_fbp", "train_fbp"} and not (args.output_dir / "vol_fbp.npy").exists() and stage_in_window(args, "init"):
            raise FileNotFoundError("FBP stage is skipped but output_dir/vol_fbp.npy is missing")

    init_stage = stage_in_window(args, "init")
    if init_stage:
        run_stage("init", build_init_command(args, init_output))
    else:
        manifest["stage_status"]["init"] = "skipped"

    train_stage = stage_in_window(args, "train")
    train_ply = artifact_for_init(args, init_output)
    if train_stage:
        if not args.dry_run and not train_ply.exists():
            raise FileNotFoundError(f"Initialization point cloud is missing: {train_ply}")
        train_source = Path(args.train_source_path or args.output_dir).resolve()
        train_model = Path(args.train_model_path or args.model_path).resolve()
        train_config_path = args.output_dir / ".pipeline_train_config.yml"
        write_train_config(args, train_config_path, train_source, train_model, Path(args.train_ply_path or init_output).resolve())
        run_stage("train", build_train_command(args, init_output, train_config_path))
    else:
        manifest["stage_status"]["train"] = "skipped"

    test_stage = stage_in_window(args, "test")
    if test_stage:
        pose_result = args.test_pose_result.resolve() if args.test_pose_result is not None else None
        if pose_result is None and args.train_pose_enabled:
            pose_model_root = Path(args.train_model_path or args.model_path).resolve()
            candidate = pose_model_root / "pose_calibration" / "pose_offsets.npz"
            if candidate.exists() or args.dry_run:
                pose_result = candidate
        if pose_result is not None and not args.dry_run and not pose_result.exists():
            raise FileNotFoundError(f"Pose result not found: {pose_result}")
        manifest["pose_result"] = pose_result
        run_stage("test", build_test_command(args, pose_result))
    else:
        manifest["stage_status"]["test"] = "skipped"
    fbp_completed = manifest["stage_status"]["fbp"] in {"success", "resumed"}
    if args.dry_run or fbp_completed or any(args.output_dir.iterdir()):
        save()
        print(f"Pipeline manifest: {manifest_path}")
    else:
        print("Pipeline stopped before FBP; no manifest was written into the fresh output directory.")


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    resolved_args = resolve_args(parsed_args)
    main(resolved_args)
