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

from __future__ import annotations

import os
import os.path as osp
import sys
import torch
from tqdm import tqdm, trange
import torchvision
from time import time
import numpy as np
import concurrent.futures
import yaml
from argparse import ArgumentParser
from random import randint

sys.path.append("./")
from r2_gaussian.arguments import (
    ModelParams,
    PipelineParams,
    get_combined_args,
)
from r2_gaussian.dataset import Scene
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.utils.general_utils import safe_state, t2a
from r2_gaussian.utils.image_utils import metric_vol, projection_metric_report
from r2_gaussian.utils.metric_report import finite_metric_values
from r2_gaussian.utils.volume_export_utils import save_nifti_xyz
from r2_gaussian.pose import StaticConePose, StaticParallelPose


def testing(
    dataset: ModelParams,
    pipeline: PipelineParams,
    iteration: int,
    skip_render_train: bool,
    skip_render_test: bool,
    skip_recon: bool,
    pose_result: str | None = None,
):
    # Set up dataset
    scene = Scene(
        dataset,
        shuffle=False,
    )

    gaussians = GaussianModel(None)
    loaded_iter = initialize_gaussian(gaussians, dataset, iteration)
    scene.gaussians = gaussians
    save_path = osp.join(
        dataset.model_path,
        "test",
        "iter_{}".format(loaded_iter),
    )
    from r2_gaussian.utils.artifact_utils import atomic_json
    status_path = osp.join(save_path, "evaluation_status.json")
    phase = "pose_setup"
    status_base = {
        "iteration": loaded_iter,
        "n_train": len(scene.getTrainCameras()),
        "n_test": len(scene.getTestCameras()),
        "render_train": not skip_render_train,
        "render_test": not skip_render_test,
        "reconstruction": not skip_recon,
    }
    atomic_json(status_path, {**status_base, "status": "running", "phase": phase})
    try:
        from r2_gaussian.pose.selection import resolve_pose_result
        pose_result = resolve_pose_result(dataset.model_path, loaded_iter, pose_result)
        pipeline.pose_enabled = pose_result is not None
        if pose_result:
            pose_result = str(pose_result)
            from r2_gaussian.pose.artifacts import validate_pose_file
            pose_indices = validate_pose_file(pose_result, scene, loaded_iter)
            scanner_mode = scene.scanner_cfg.get("mode")
            if scanner_mode == "parallel":
                static_pose = StaticParallelPose.from_npz(pose_result)
                values = static_pose.values()
                # Only angle/per-view-shift arrays encode the number of views;
                # global shift, beam tilt and roll have lengths 2/1.
                per_view = [
                    value for key, value in values.items()
                    if key in ("delta_angle", "view_shift") and value is not None
                ]
                if per_view and any(
                    value.shape[0] != len(scene.train_cameras) for value in per_view
                ):
                    raise ValueError(
                        "Pose result per-view arrays do not match the number of training "
                        f"{len(scene.train_cameras)} training views."
                    )
                for pose_index, camera in enumerate(scene.train_cameras):
                    camera.attach_parallel_pose(static_pose, pose_indices[pose_index])
                # Shared corrections also apply to held-out views. Train-only
                # per-view angle/shift arrays do not use unrelated test indices.
                for camera in scene.test_cameras:
                    camera.attach_parallel_pose(static_pose, None)
            elif scanner_mode == "cone":
                static_pose = StaticConePose.from_npz(pose_result)
                values = static_pose.values()
                per_view = [value for value in values.values() if value is not None]
                if per_view and any(
                    value.shape[0] != len(scene.train_cameras) for value in per_view
                ):
                    raise ValueError(
                        "Pose result per-view arrays do not match the number of training "
                        f"{len(scene.train_cameras)} training views."
                    )
                for pose_index, camera in enumerate(scene.train_cameras):
                    camera.attach_cone_pose(static_pose, pose_indices[pose_index])
            else:
                raise ValueError(f"Unsupported scanner mode: {scanner_mode!r}")
        phase = "backend_validation"
        atomic_json(status_path, {**status_base, "status": "running", "phase": phase})
        from r2_gaussian.gaussian.render_query import raster_backend_provenance
        native_build = raster_backend_provenance(getattr(pipeline, "pose_enabled", False))

        phase = "render_train"
        atomic_json(status_path, {**status_base, "status": "running", "phase": phase})
        if not skip_render_train and scene.getTrainCameras():
            evaluate_render(save_path, "render_train", scene.getTrainCameras(), gaussians, pipeline)

        phase = "render_test"
        atomic_json(status_path, {**status_base, "status": "running", "phase": phase})
        if not skip_render_test and scene.getTestCameras():
            evaluate_render(save_path, "render_test", scene.getTestCameras(), gaussians, pipeline)
        if not skip_render_test and not scene.getTestCameras():
            os.makedirs(save_path, exist_ok=True)
            with open(osp.join(save_path, "eval2d_render_test.yml"), "w") as stream:
                yaml.safe_dump({"status": "skipped", "reason": "no test views", "n_views": 0}, stream)

        phase = "reconstruction"
        atomic_json(status_path, {**status_base, "status": "running", "phase": phase})
        if not skip_recon:
            evaluate_volume(
                save_path,
                "reconstruction",
                scene.scanner_cfg,
                gaussians,
                pipeline,
                scene.vol_gt,
                scene_scale=scene.scene_scale,
            )
        atomic_json(status_path, {
            **status_base, "status": "completed", "native_build": native_build,
        })
    except Exception as error:
        atomic_json(status_path, {
            **status_base,
            "status": "failed",
            "phase": phase,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise


def evaluate_volume(
    save_path,
    name,
    scanner_cfg,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    vol_gt,
    scene_scale=1.0,
):
    """Evaluate volume reconstruction."""
    slice_save_path = osp.join(save_path, name)
    os.makedirs(slice_save_path, exist_ok=True)

    query_pkg = query(
        gaussians,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipeline,
    )
    vol_pred = query_pkg["vol"]

    psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
    ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")

    multithread_write(
        [vol_gt[..., i][None] for i in range(vol_gt.shape[2])],
        slice_save_path,
        "_gt",
        display_xy=True,
    )
    multithread_write(
        [vol_pred[..., i][None] for i in range(vol_pred.shape[2])],
        slice_save_path,
        "_pred",
        display_xy=True,
    )
    eval_dict = {
        "psnr_3d": psnr_3d,
        "ssim_3d": ssim_3d,
        "ssim_3d_x": ssim_3d_axis[0],
        "ssim_3d_y": ssim_3d_axis[1],
        "ssim_3d_z": ssim_3d_axis[2],
        "volume_metric_version": "all_slices_v2",
    }

    with open(osp.join(save_path, "eval3d.yml"), "w") as f:
        yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)

    np.save(osp.join(save_path, "vol_gt.npy"), t2a(vol_gt))
    np.save(osp.join(save_path, "vol_pred.npy"), t2a(vol_pred))
    # Restore dataset lengths; leave density values and query coordinates intact.
    export_scanner = {
        "sVoxel": (np.asarray(scanner_cfg["sVoxel"]) / scene_scale).tolist(),
        "offOrigin": (np.asarray(scanner_cfg["offOrigin"]) / scene_scale).tolist(),
    }
    save_nifti_xyz(t2a(vol_gt), export_scanner, osp.join(save_path, "vol_gt.nii.gz"))
    save_nifti_xyz(t2a(vol_pred), export_scanner, osp.join(save_path, "vol_pred.nii.gz"))

    print(f"{name} complete. psnr_3d: {psnr_3d}, ssim_3d: {ssim_3d}")


def evaluate_render(save_path, name, views, gaussians, pipeline):
    """Evaluate projection rendering."""
    proj_save_path = osp.join(save_path, name)

    # If already rendered, skip.
    if osp.exists(osp.join(save_path, "eval.yml")):
        print("{} in {} already rendered. Skip.".format(name, save_path))
        return
    os.makedirs(proj_save_path, exist_ok=True)

    gt_list = []
    render_list = []
    for view in tqdm(views, desc="render {}".format(name), leave=False):
        rendering = render(view, gaussians, pipeline)["render"]
        gt = view.original_image[0:3, :, :]
        gt_list.append(gt)
        render_list.append(rendering)
    multithread_write(gt_list, proj_save_path, "_gt")
    multithread_write(render_list, proj_save_path, "_pred")

    images = torch.concat(render_list, 0).permute(1, 2, 0)
    gt_images = torch.concat(gt_list, 0).permute(1, 2, 0)
    eval_dict = projection_metric_report(gt_images, images,
                                        data_range=getattr(pipeline, "projection_data_range", 0.0) or None)
    with open(osp.join(save_path, f"eval2d_{name}.yml"), "w") as f:
        yaml.safe_dump(finite_metric_values(eval_dict), f, default_flow_style=False, sort_keys=False)
    print(
        f"{name} complete. psnr_2d: {eval_dict['psnr_2d']}, ssim_2d: {eval_dict['ssim_2d']}."
    )


def multithread_write(image_list, path, suffix, display_xy=False):
    def write_image(image, count, path):
        torchvision.utils.save_image(
            torch.flip(image.transpose(-2, -1), dims=(-2,)) if display_xy else image,
            osp.join(path, "{0:05d}".format(count) + "{}.png".format(suffix))
        )
        np.save(
            osp.join(path, "{0:05d}".format(count) + "{}.npy".format(suffix)),
            image.cpu().numpy()[0],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        tasks = [executor.submit(write_image, image, index, path)
                 for index, image in enumerate(image_list)]
        for task in tasks:
            # Surface worker errors before marking evaluation as completed.
            task.result()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser, sentinel=True)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_render_train", action="store_true", default=False)
    parser.add_argument("--skip_render_test", action="store_true", default=False)
    parser.add_argument("--skip_recon", action="store_true", default=False)
    parser.add_argument(
        "--pose_result",
        type=str,
        default=None,
        help="Optional pose_calibration/pose_offsets.npz; per-view corrections apply to training views, shared corrections also apply to held-out views.",
    )
    args = get_combined_args(parser)

    safe_state(args.quiet)

    with torch.no_grad():
        testing(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            args.skip_render_train,
            args.skip_render_test,
            args.skip_recon,
            getattr(args, "pose_result", None),
        )
