#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import os.path as osp
import csv
import json
import math
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml

sys.path.append("./")
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, query_points, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state
from r2_gaussian.utils.cfg_utils import load_config
from r2_gaussian.utils.log_utils import prepare_output_and_logger
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from r2_gaussian.utils.image_utils import metric_vol, projection_metric_report
from r2_gaussian.utils.metric_report import finite_metric_values
from r2_gaussian.utils.plot_utils import show_two_slice as _show_two_slice


def show_volume_slices(slice1, slice2, *args, **kwargs):
    """Keep training previews in the same display coordinates as XYZ exports."""
    first = slice1.detach().cpu().numpy() if torch.is_tensor(slice1) else np.asarray(slice1)
    second = slice2.detach().cpu().numpy() if torch.is_tensor(slice2) else np.asarray(slice2)
    return _show_two_slice(np.flipud(first.T), np.flipud(second.T), *args, **kwargs)
from r2_gaussian.pose import (
    PARALLEL_POSE_FORMAT_VERSION,
    ParallelPose,
    ConePose,
)


def _prepare_slice_supervision(opt, scanner_cfg, device):
    """Load and validate the fixed reference volume for sparse slice loss."""
    if not getattr(opt, "slice_loss_enabled", False):
        return None
    if getattr(opt, "pose_enabled", False):
        raise ValueError("slice_loss_enabled is not supported with pose_enabled")
    reference_path = getattr(opt, "slice_reference_path", "")
    if not reference_path:
        raise ValueError("slice_reference_path is required when slice_loss_enabled=true")
    if not osp.exists(reference_path):
        raise FileNotFoundError(f"Slice reference volume not found: {reference_path}")
    reference = np.asarray(np.load(reference_path, mmap_mode="r"), dtype=np.float32)
    expected_shape = tuple(int(value) for value in scanner_cfg["nVoxel"])
    if reference.ndim != 3 or tuple(reference.shape) != expected_shape:
        raise ValueError(
            f"Slice reference shape {reference.shape} does not match scanner nVoxel {expected_shape}"
        )
    if not np.isfinite(reference).all():
        raise ValueError("Slice reference volume contains non-finite values")
    axis = str(getattr(opt, "slice_axis", "z")).lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("slice_axis must be one of x, y, z")
    interval = int(getattr(opt, "slice_interval", 50))
    if interval < 1:
        raise ValueError("slice_interval must be at least 1")
    if int(getattr(opt, "slice_end_iter", 15000)) < int(
        getattr(opt, "slice_start_iter", 500)
    ):
        raise ValueError("slice_end_iter must be greater than or equal to slice_start_iter")
    ratio = float(getattr(opt, "slice_filter_ratio", 10.0))
    if not (0.0 < ratio <= 100.0):
        raise ValueError("slice_filter_ratio must be in (0, 100]")
    max_points = int(getattr(opt, "slice_max_points", 200_000))
    if max_points < 1:
        raise ValueError("slice_max_points must be positive")
    threshold = getattr(opt, "slice_filter_threshold", 0.01)
    if threshold is not None and not np.isfinite(float(threshold)):
        raise ValueError("slice_filter_threshold must be finite or null")
    if float(getattr(opt, "lambda_slice", 0.05)) < 0:
        raise ValueError("lambda_slice must be non-negative")
    if int(getattr(opt, "slice_query_chunk_size", 1024)) < 1:
        raise ValueError("slice_query_chunk_size must be positive")
    return {
        "volume": reference,
        "axis": axis,
        "axis_index": {"x": 0, "y": 1, "z": 2}[axis],
        "generator": torch.Generator().manual_seed(
            int(getattr(opt, "slice_seed", 0))
        ),
        "threshold": None if threshold is None else float(threshold),
        "ratio": ratio,
        "max_points": max_points,
        "device": device,
    }


def _sample_sparse_slice(supervision, iteration, opt, scanner_cfg):
    """Return sparse world points and reference values, or None if skipped."""
    start = int(getattr(opt, "slice_start_iter", 500))
    end = int(getattr(opt, "slice_end_iter", 15000))
    interval = int(getattr(opt, "slice_interval", 50))
    if iteration < start or iteration > end or (iteration - start) % interval:
        return None
    volume = supervision["volume"]
    axis = supervision["axis_index"]
    n_voxel = np.asarray(scanner_cfg["nVoxel"], dtype=np.int64)
    n_slices = int(n_voxel[axis])
    slice_index = int(torch.randint(n_slices, (1,), generator=supervision["generator"]).item())
    reference_slice = np.take(volume, slice_index, axis=axis)
    threshold = supervision["threshold"]
    if threshold is None:
        candidate = np.arange(reference_slice.size, dtype=np.int64)
    else:
        candidate = np.flatnonzero(reference_slice.reshape(-1) > threshold)
    if candidate.size == 0:
        return {"skipped": True, "axis": axis, "slice_index": slice_index, "candidates": 0}
    count = max(1, int(np.ceil(candidate.size * supervision["ratio"] / 100.0)))
    count = min(count, supervision["max_points"], candidate.size)
    order = torch.randperm(candidate.size, generator=supervision["generator"])[:count].numpy()
    chosen = candidate[order]
    ref_values = reference_slice.reshape(-1)[chosen].astype(np.float32, copy=False)
    plane_shape = [int(n_voxel[i]) for i in range(3) if i != axis]
    plane_indices = np.column_stack(np.unravel_index(chosen, tuple(plane_shape)))
    indices = np.zeros((count, 3), dtype=np.float32)
    other_axes = [i for i in range(3) if i != axis]
    indices[:, axis] = slice_index
    for col, coord_axis in enumerate(other_axes):
        indices[:, coord_axis] = plane_indices[:, col]
    s_voxel = np.asarray(scanner_cfg["sVoxel"], dtype=np.float32)
    off_origin = np.asarray(scanner_cfg["offOrigin"], dtype=np.float32)
    d_voxel = s_voxel / n_voxel.astype(np.float32)
    points_world = off_origin - s_voxel / 2.0 + (indices + 0.5) * d_voxel
    return {
        "points_world": torch.from_numpy(points_world).to(supervision["device"]),
        "reference": torch.from_numpy(ref_values).to(supervision["device"]),
        "axis": axis,
        "slice_index": slice_index,
        "candidates": int(candidate.size),
        "count": count,
    }


def _shape_regularization(gaussians, opt, iteration):
    """Return thresholded shape penalties and diagnostics.

    The branch is deliberately opt-in.  Ratios below the configured limits
    have exactly zero penalty, so this does not push ordinary Gaussians toward
    spherical shapes.
    """
    if not getattr(opt, "shape_enabled", False):
        return None, {}
    zero = gaussians.get_xyz.new_zeros(())
    start = int(getattr(opt, "shape_start_iter", 500))
    end = int(getattr(opt, "shape_end_iter", 15000))
    if iteration < start or iteration > end:
        return zero, {}
    scales = gaussians.get_scaling
    ordered = torch.sort(scales, dim=1).values.clamp_min(1e-12)
    needle_ratio = ordered[:, 2] / ordered[:, 1]
    sheet_ratio = ordered[:, 1] / ordered[:, 0]
    excess_needle = torch.relu(
        torch.log(needle_ratio) - np.log(float(getattr(opt, "shape_needle_threshold", 10.0)))
    )
    excess_sheet = torch.relu(
        torch.log(sheet_ratio) - np.log(float(getattr(opt, "shape_sheet_threshold", 10.0)))
    )
    ramp_iters = max(1, int(getattr(opt, "shape_ramp_iters", 1000)))
    ramp = min(1.0, max(0.0, (iteration - start + 1) / ramp_iters))
    needle_loss = excess_needle.square().mean() if getattr(opt, "shape_needle_enabled", True) else zero
    sheet_loss = excess_sheet.square().mean() if getattr(opt, "shape_sheet_enabled", True) else zero
    total = ramp * (
        float(getattr(opt, "lambda_shape_needle", 0.0)) * needle_loss
        + float(getattr(opt, "lambda_shape_sheet", 0.0)) * sheet_loss
    )
    return total, {
        "shape_needle": needle_loss,
        "shape_sheet": sheet_loss,
        "shape_needle_count": int((needle_ratio > float(getattr(opt, "shape_needle_threshold", 10.0))).sum().item()),
        "shape_sheet_count": int((sheet_ratio > float(getattr(opt, "shape_sheet_threshold", 10.0))).sum().item()),
        "shape_needle_p99": float(torch.quantile(needle_ratio.detach(), 0.99).item()),
        "shape_sheet_p99": float(torch.quantile(sheet_ratio.detach(), 0.99).item()),
    }


def _validate_shape_options(opt):
    if not getattr(opt, "shape_enabled", False):
        return
    for name in ("shape_needle_threshold", "shape_sheet_threshold",
                 "shape_hard_needle_ratio", "shape_hard_sheet_ratio"):
        value = float(getattr(opt, name, 0.0))
        if not np.isfinite(value) or value <= 1.0:
            raise ValueError(f"{name} must be finite and greater than 1")
    for name in ("lambda_shape_needle", "lambda_shape_sheet",
                 "shape_low_density_threshold", "shape_low_mass_threshold"):
        value = float(getattr(opt, name, 0.0))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not (getattr(opt, "shape_needle_enabled", False) or getattr(opt, "shape_sheet_enabled", False)):
        raise ValueError("Enable at least one of shape_needle_enabled or shape_sheet_enabled")
    if int(getattr(opt, "shape_end_iter", 0)) < int(getattr(opt, "shape_start_iter", 0)):
        raise ValueError("shape_end_iter must be greater than or equal to shape_start_iter")
    if int(getattr(opt, "shape_check_interval", 0)) < 1 or int(getattr(opt, "shape_patience", 0)) < 1:
        raise ValueError("shape_check_interval and shape_patience must be positive")
    if int(getattr(opt, "shape_ramp_iters", 0)) < 0:
        raise ValueError("shape_ramp_iters must be non-negative")
    if int(getattr(opt, "shape_split_count", 0)) < 2:
        raise ValueError("shape_split_count must be at least 2")
    density_gain = float(getattr(opt, "shape_repair_max_density_gain", 0.0))
    if not np.isfinite(density_gain) or density_gain < 1.0:
        raise ValueError("shape_repair_max_density_gain must be at least 1")
    if int(getattr(opt, "shape_max_hard_actions_per_check", 0)) < 1:
        raise ValueError("shape_max_hard_actions_per_check must be positive")
    if str(getattr(opt, "shape_hard_action", "none")) not in {"none", "prune_low", "repair", "split", "auto"}:
        raise ValueError("shape_hard_action must be none, prune_low, repair, split, or auto")


def _check_model_finite(gaussians, iteration, stage):
    """Identify the first failing training stage without masking bad values."""
    if gaussians.get_xyz.shape[0] == 0:
        raise ValueError(f"[ITER {iteration}] Empty Gaussian model after {stage}")
    invalid = []
    for name in ("_xyz", "_scaling", "_rotation", "_density"):
        value = getattr(gaussians, name)
        bad = ~torch.isfinite(value)
        if bad.any():
            invalid.append(f"{name}: {int(bad.sum().item())}/{value.numel()} nonfinite values")
    if invalid:
        raise ValueError(
            f"[ITER {iteration}] Nonfinite Gaussian model after {stage}: " + "; ".join(invalid)
        )


def _shape_hard_process(gaussians, opt, iteration):
    """Update persistence counters and optionally prune/repair/split anomalies."""
    if not getattr(opt, "shape_enabled", False):
        return {}
    interval = max(1, int(getattr(opt, "shape_check_interval", 100)))
    if iteration < int(getattr(opt, "shape_start_iter", 500)) or iteration % interval:
        return {}
    if iteration > int(getattr(opt, "shape_end_iter", 15000)):
        return {}
    if gaussians.shape_bad_count is None or gaussians.shape_bad_count.shape[0] != gaussians.get_xyz.shape[0]:
        gaussians.enable_shape_tracking()
    scales = gaussians.get_scaling.detach().clamp_min(1e-12)
    ordered = torch.sort(scales, dim=1).values
    needle = ordered[:, 2] / ordered[:, 1] > float(getattr(opt, "shape_hard_needle_ratio", 50.0))
    sheet = ordered[:, 1] / ordered[:, 0] > float(getattr(opt, "shape_hard_sheet_ratio", 50.0))
    severe = (needle & bool(getattr(opt, "shape_needle_enabled", True))) | (
        sheet & bool(getattr(opt, "shape_sheet_enabled", True))
    )
    gaussians.shape_bad_count = torch.where(
        severe, gaussians.shape_bad_count + 1, torch.zeros_like(gaussians.shape_bad_count)
    )
    persistent = gaussians.shape_bad_count >= max(1, int(getattr(opt, "shape_patience", 3)))
    action = str(getattr(opt, "shape_hard_action", "none"))
    if action == "none" or not persistent.any():
        return {"shape_persistent": int(persistent.sum().item())}
    candidates = torch.nonzero(persistent, as_tuple=False).flatten()
    limit = max(1, int(getattr(opt, "shape_max_hard_actions_per_check", 32)))
    if candidates.numel() > limit:
        candidates = candidates[:limit]
    candidate_mask = torch.zeros_like(persistent)
    candidate_mask[candidates] = True
    density = gaussians.get_density.detach().reshape(-1)
    mass = density * torch.prod(scales, dim=1) * ((2.0 * np.pi) ** 1.5)
    low = torch.zeros_like(candidate_mask)
    density_limit = float(getattr(opt, "shape_low_density_threshold", 0.0))
    mass_limit = float(getattr(opt, "shape_low_mass_threshold", 0.0))
    if density_limit > 0:
        low |= density < density_limit
    if mass_limit > 0:
        low |= mass < mass_limit
    low &= candidate_mask
    pruned = repaired = split = 0
    if action in ("repair", "auto"):
        target = candidate_mask & ~low
        repaired = gaussians.shape_repair(
            target,
            float(getattr(opt, "shape_needle_threshold", 10.0)),
            float(getattr(opt, "shape_sheet_threshold", 10.0)),
            float(getattr(opt, "shape_repair_max_density_gain", 4.0)),
            bool(getattr(opt, "shape_needle_enabled", True)),
            bool(getattr(opt, "shape_sheet_enabled", True)),
        )
    if action in ("prune_low", "auto") and low.any():
        if low.all():
            # Keep the highest-mass point rather than deleting the model.
            low[torch.argmax(mass)] = False
        gaussians.prune_points(low)
        pruned = int(low.sum().item())
    elif action == "split":
        children = int(getattr(opt, "shape_split_count", 2))
        cap = int(getattr(opt, "max_num_gaussians", 0) or 0)
        if cap > 0:
            available = max(0, cap - gaussians.get_xyz.shape[0])
            max_parents = available // max(1, children - 1)
            split_candidates = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            if split_candidates.numel() > max_parents:
                candidate_mask.zero_()
                if max_parents:
                    candidate_mask[split_candidates[:max_parents]] = True
        split = gaussians.shape_split(candidate_mask, children)
    return {
        "shape_persistent": int(persistent.sum().item()),
        "shape_pruned": pruned,
        "shape_repaired": repaired,
        "shape_split": split,
    }


def save_pose_result(scene, pose_model, iteration=None):
    """Save learned pose offsets without touching the input dataset."""
    from r2_gaussian.utils.artifact_utils import atomic_npz
    output_dir = osp.join(scene.model_path, "pose_calibration")
    os.makedirs(output_dir, exist_ok=True)
    values = {
        key: value.detach().cpu().numpy()
        for key, value in pose_model.values().items()
        if value is not None
    }
    if iteration is not None:
        from r2_gaussian.pose.artifacts import pose_metadata
        from r2_gaussian.utils.artifact_utils import atomic_json
        values.update(pose_metadata(scene, iteration))
        snapshot = osp.join(output_dir, f"iteration_{iteration}")
        os.makedirs(snapshot, exist_ok=True)
        snapshot_values = dict(values)
        if scene.scanner_cfg["mode"] == "parallel":
            snapshot_values["pose_format_version"] = np.asarray(PARALLEL_POSE_FORMAT_VERSION, dtype=np.int32)
        from r2_gaussian.utils.artifact_utils import atomic_npz
        atomic_npz(osp.join(snapshot, "pose_offsets.npz"), **snapshot_values)
        atomic_json(osp.join(output_dir, "pose_required.json"), {"latest_iteration": iteration})
    # Cone-beam uses the target repository's translation/quaternion names.
    if "delta_translation" in values or "delta_quaternion" in values:
        atomic_npz(osp.join(output_dir, "pose_offsets.npz"), **values)
        atomic_npz(
            osp.join(output_dir, "cone_pose_offsets.npz"),
            **values,
        )
        return

    atomic_npz(
        osp.join(output_dir, "pose_offsets.npz"),
        pose_format_version=np.asarray(PARALLEL_POSE_FORMAT_VERSION, dtype=np.int32),
        **values,
    )

    delta_angle = values.get("delta_angle")
    view_shift = values.get("view_shift")
    global_shift = values.get("global_shift", np.zeros(2, dtype=np.float32))
    angle_values = (
        delta_angle
        if delta_angle is not None
        else np.zeros(len(scene.train_cameras), dtype=np.float32)
    )
    view_values = (
        view_shift
        if view_shift is not None
        else np.zeros((len(scene.train_cameras), 2), dtype=np.float32)
    )
    beam_values = values.get("beam_tilt", np.zeros(2, dtype=np.float32))
    detector_roll = values.get("detector_roll", np.zeros(1, dtype=np.float32))

    with open(osp.join(output_dir, "pose_report.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "view",
            "image",
            "angle_delta_rad",
            "shift_u_px",
            "shift_v_px",
            "beam_tilt_u_rad",
            "beam_tilt_v_rad",
            "detector_roll_rad",
        ])
        for index, camera in enumerate(scene.train_cameras):
            writer.writerow([
                index,
                camera.image_name,
                float(angle_values[index]),
                float(global_shift[0] + view_values[index, 0]),
                float(global_shift[1] + view_values[index, 1]),
                float(beam_values[0]),
                float(beam_values[1]),
                float(detector_roll[0]),
            ])


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
):
    first_iter = 0

    # Set up dataset
    from r2_gaussian.gaussian.render_query import raster_backend_provenance
    native_build = raster_backend_provenance(bool(getattr(opt, "pose_enabled", False)))
    scene = Scene(dataset, shuffle=False)
    # The renderer receives this runtime-only flag through PipelineParams.
    pipe.pose_enabled = bool(getattr(opt, "pose_enabled", False))

    # Optional pose-calibration branch.  It is completely inactive unless
    # pose_enabled is explicitly set in the config/CLI.
    pose_model = None
    pose_optimizer = None
    if getattr(opt, "pose_enabled", False):
        scanner_mode = scene.scanner_cfg.get("mode", "parallel")
        pose_mode = getattr(opt, "pose_mode", "auto")
        if pose_mode not in ("auto", scanner_mode):
            raise ValueError(
                f"pose_mode={pose_mode!r} does not match scanner mode={scanner_mode!r}"
            )
        if scanner_mode == "parallel":
            pose_cfg = {
                "optimize_angle": getattr(opt, "parallel_optimize_angle", False),
                "optimize_global_detector_shift": getattr(
                    opt, "parallel_optimize_global_detector_shift", False
                ),
                "optimize_per_view_detector_shift": getattr(
                    opt, "parallel_optimize_per_view_detector_shift", False
                ),
                "optimize_beam_tilt": getattr(
                    opt, "parallel_optimize_beam_tilt", False
                ),
                "optimize_detector_roll": getattr(
                    opt, "parallel_optimize_detector_roll", False
                ),
                "angle_max_deg": getattr(opt, "parallel_angle_max_deg", 0.5),
                "detector_shift_max_pixel": getattr(
                    opt, "parallel_detector_shift_max_pixel", 3.0
                ),
                "beam_tilt_max_deg": getattr(
                    opt, "parallel_beam_tilt_max_deg", 0.5
                ),
                "detector_roll_max_deg": getattr(
                    opt, "parallel_detector_roll_max_deg", 0.5
                ),
            }
            pose_model = ParallelPose(
                len(scene.train_cameras), pose_cfg, device="cuda"
            )
        elif scanner_mode == "cone":
            pose_model = ConePose(
                len(scene.train_cameras),
                {
                    "optimize_translation": getattr(
                        opt, "cone_optimize_translation", True
                    ),
                    "optimize_quaternion": getattr(
                        opt, "cone_optimize_quaternion", True
                    ),
                },
                device="cuda",
            )
        else:
            raise ValueError(f"Unsupported scanner mode: {scanner_mode!r}")

        trainable_pose_params = pose_model.optimizable_parameters()
        if trainable_pose_params:
            pose_optimizer = torch.optim.Adam(
                trainable_pose_params,
                lr=getattr(opt, "pose_lr", 1.0e-4),
            )
        for pose_index, camera in enumerate(scene.train_cameras):
            if scanner_mode == "parallel":
                camera.attach_parallel_pose(pose_model, pose_index)
            else:
                camera.attach_cone_pose(pose_model, pose_index)
        if scanner_mode == "parallel":
            # Held-out views receive only parameters shared across views;
            # per-view angle and detector shifts are learned for train views.
            for camera in scene.test_cameras:
                camera.attach_parallel_pose(pose_model, None)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world
    queryfunc = lambda x: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
    )

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound)
    scene.gaussians = gaussians
    if checkpoint is None:
        initialize_gaussian(gaussians, dataset, None)
        gaussians.training_setup(opt)
    pose_checkpoint_state = None
    pose_optimizer_checkpoint_state = None
    if checkpoint is not None:
        from r2_gaussian.utils.artifact_utils import load_training_checkpoint
        checkpoint_data = load_training_checkpoint(checkpoint)
        if len(checkpoint_data) == 2:
            raise ValueError("Unbound legacy Gaussian checkpoint: dataset/geometry cannot be verified; use a bound checkpoint or start a fresh run")
        elif len(checkpoint_data) == 3:
            model_params, first_iter, checkpoint_binding = checkpoint_data
            from r2_gaussian.pose.artifacts import validate_gaussian_checkpoint
            validate_gaussian_checkpoint(checkpoint_binding, scene, first_iter)
        elif len(checkpoint_data) == 5:
            (
                model_params,
                first_iter,
                pose_checkpoint_state,
                pose_optimizer_checkpoint_state,
                checkpoint_binding,
            ) = checkpoint_data
            from r2_gaussian.pose.artifacts import validate_checkpoint_binding
            validate_checkpoint_binding(checkpoint_binding, scene, first_iter)
        elif len(checkpoint_data) == 4:
            raise ValueError("Unbound legacy pose checkpoint: start fresh or export a bound checkpoint")
        else:
            raise ValueError(
                "Unsupported checkpoint format: expected a Gaussian checkpoint "
                "with 3 entries or a bound pose checkpoint with 5 entries."
            )
        if pose_checkpoint_state is not None and pose_model is None:
            raise ValueError("Pose-aware checkpoint requires pose_enabled; refusing to discard learned geometry")
        gaussians.restore(model_params, opt)
        if pose_model is not None:
            if pose_checkpoint_state is None:
                raise ValueError(
                    "Cannot resume pose-enabled training from a Gaussian-only "
                    "checkpoint; its learned camera geometry was not saved. "
                    "Start a fresh training run or use a pose-aware checkpoint."
                )
            pose_model.load_state_dict(pose_checkpoint_state)
            if pose_optimizer is not None:
                if pose_optimizer_checkpoint_state is None:
                    raise ValueError(
                        "Pose-aware checkpoint is missing pose optimizer state; "
                        "resume would reset the geometry optimizer."
                    )
                pose_optimizer.load_state_dict(pose_optimizer_checkpoint_state)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    _validate_shape_options(opt)
    if getattr(opt, "shape_enabled", False) and gaussians.shape_bad_count is None:
        gaussians.enable_shape_tracking()

    slice_supervision = _prepare_slice_supervision(
        opt, scanner_cfg, gaussians.get_xyz.device
    )

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        shape_diag = {}
        shape_action_diag = {}

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        render_pkg = render(viewpoint_cam, gaussians, pipe)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        # Optional anomalous-shape suppression.  This branch is completely
        # inactive by default, preserving the historical loss path.
        if getattr(opt, "shape_enabled", False):
            shape_loss, shape_diag = _shape_regularization(gaussians, opt, iteration)
        else:
            shape_loss, shape_diag = None, {}
        if shape_diag:
            loss["shape_needle"] = shape_diag["shape_needle"]
            loss["shape_sheet"] = shape_diag["shape_sheet"]
            loss["total"] = loss["total"] + shape_loss

        # Optional sparse initialization-volume slice supervision. Only the
        # sampled reference positions are queried, so no dense slice volume is
        # allocated and the branch remains inactive by default.
        if slice_supervision is not None:
            sampled_slice = _sample_sparse_slice(
                slice_supervision, iteration, opt, scanner_cfg
            )
            if sampled_slice is not None:
                if sampled_slice.get("skipped", False):
                    tqdm.write(
                        f"[ITER {iteration}] Slice loss skipped: no reference values above threshold"
                    )
                else:
                    prediction = query_points(
                        gaussians,
                        sampled_slice["points_world"],
                        pipe,
                        chunk_size=getattr(opt, "slice_query_chunk_size", 1024),
                    )
                    loss_slice = torch.abs(
                        prediction - sampled_slice["reference"]
                    ).mean()
                    loss["slice"] = loss_slice
                    loss["total"] = (
                        loss["total"]
                        + getattr(opt, "lambda_slice", 0.05) * loss_slice
                    )
                    tqdm.write(
                        f"[ITER {iteration}] Sparse slice axis={sampled_slice['axis']} "
                        f"index={sampled_slice['slice_index']} "
                        f"candidates={sampled_slice['candidates']} "
                        f"samples={sampled_slice['count']}"
                    )

        if pose_model is not None:
            loss_pose = pose_model.regularization()
            loss["pose_smooth"] = loss_pose
            pose_weight = (
                getattr(opt, "pose_lambda_magnitude", 0.0)
                if scene.scanner_cfg.get("mode") == "cone"
                else getattr(opt, "pose_lambda_smooth", 0.1)
            )
            loss["total"] = (
                loss["total"]
                + pose_weight * loss_pose
            )

        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            # Step while gradients still belong to the live Parameters.
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if pose_optimizer is not None:
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)
            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if getattr(opt, "shape_enabled", False):
                _check_model_finite(gaussians, iteration, "adaptive densification/pruning")

            # Hard actions are checked only at the configured cadence and
            # after the optimizer step, so no live autograd graph references
            # parameters that are being pruned/replaced.
            if getattr(opt, "shape_enabled", False):
                _check_model_finite(gaussians, iteration, "optimizer step")
                shape_action_diag = _shape_hard_process(gaussians, opt, iteration)
                if getattr(opt, "shape_hard_action", "none") != "none":
                    _check_model_finite(gaussians, iteration, "shape hard processing")

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)
                if pose_model is not None:
                    save_pose_result(scene, pose_model, iteration)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                checkpoint_data = (gaussians.capture(), iteration)
                if pose_model is not None:
                    from r2_gaussian.pose.artifacts import pose_metadata
                    checkpoint_data += (
                        pose_model.state_dict(),
                        pose_optimizer.state_dict() if pose_optimizer is not None else None,
                        pose_metadata(scene, iteration),
                    )
                else:
                    from r2_gaussian.pose.artifacts import gaussian_checkpoint_metadata
                    checkpoint_data += (gaussian_checkpoint_metadata(scene, iteration),)
                torch.save(
                    checkpoint_data,
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = loss[l].item()
            for key, value in shape_diag.items():
                if key.endswith("_count") or key.endswith("_p99"):
                    metrics[key] = float(value)
            for key, value in shape_action_diag.items():
                metrics[key] = float(value)
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render(x, y, pipe),
                queryfunc,
                projection_data_range=getattr(pipe, "projection_data_range", 0.0) or None,
            )

    if pose_model is not None:
        save_pose_result(scene, pose_model, opt.iterations)

    final_num_gaussians = int(gaussians.get_xyz.shape[0])
    summary_path = osp.join(scene.model_path, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "iteration": opt.iterations,
                "num_gaussians": final_num_gaussians,
                "max_num_gaussians": opt.max_num_gaussians,
                "native_build": native_build,
            },
            summary_file,
            indent=2,
        )
        summary_file.write("\n")
    print(f"Final Gaussian count: {final_num_gaussians:,}")
    print(f"Training summary saved to: {summary_path}")


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
    projection_data_range=None,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                _show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                eval_dict_2d = projection_metric_report(gt_images, images, data_range=projection_data_range)
                psnr_2d, ssim_2d = eval_dict_2d["psnr_2d"], eval_dict_2d["ssim_2d"]
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        finite_metric_values(eval_dict_2d), f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    if math.isfinite(psnr_2d):
                        tb_writer.add_scalar(config["name"] + "/psnr_2d", psnr_2d, iteration)
                    tb_writer.add_scalar(config["name"] + "/perfect_view_count",
                                         eval_dict_2d["perfect_view_count"], iteration)
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        vol_pred = queryFunc(scene.gaussians)["vol"]
        vol_gt = scene.vol_gt
        psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
        ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
        eval_dict = {
            "psnr_3d": psnr_3d,
            "ssim_3d": ssim_3d,
            "ssim_3d_x": ssim_3d_axis[0],
            "ssim_3d_y": ssim_3d_axis[1],
            "ssim_3d_z": ssim_3d_axis[2],
            "volume_metric_version": "all_slices_v2",
        }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            image_show_3d = np.concatenate(
                [
                    show_volume_slices(
                        vol_gt[..., i],
                        vol_pred[..., i],
                        f"slice {i} gt",
                        f"slice {i} pred",
                        vmin=vol_gt[..., i].min(),
                        vmax=vol_gt[..., i].max(),
                        save=True,
                    )
                    for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                ],
                axis=0,
            )
            image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
            tb_writer.add_images(
                "reconstruction/slice-gt_pred_diff",
                image_show_3d,
                global_step=iteration,
            )
            tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
            tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    # fmt: on

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )

    # All done
    print("Training complete.")
