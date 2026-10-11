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
import sys
import torch
import math
from pathlib import Path


def _load_raster_backend(pose_enabled):
    if pose_enabled:
        local_backend = (
            Path(__file__).resolve().parents[1]
            / "submodules"
            / "self-xray-gaussian-rasterization-voxelization"
        )
        if str(local_backend) not in sys.path:
            sys.path.insert(0, str(local_backend))
        try:
            from self_xray_gaussian_rasterization_voxelization import (
                GaussianRasterizationSettings,
                GaussianRasterizer,
                GaussianVoxelizationSettings,
                GaussianVoxelizer,
            )
            return (
                GaussianRasterizationSettings,
                GaussianRasterizer,
                GaussianVoxelizationSettings,
                GaussianVoxelizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Pose calibration requires the target repository's "
                "self_xray_gaussian_rasterization_voxelization CUDA extension. "
                "Build/install that extension before enabling pose_enabled."
            ) from exc

    from xray_gaussian_rasterization_voxelization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        GaussianVoxelizationSettings,
        GaussianVoxelizer,
    )
    return (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        GaussianVoxelizationSettings,
        GaussianVoxelizer,
    )

sys.path.append("./")
from r2_gaussian.gaussian.gaussian_model import GaussianModel
from r2_gaussian.dataset.cameras import Camera
from r2_gaussian.arguments import PipelineParams


def query(
    pc: GaussianModel,
    center,
    nVoxel,
    sVoxel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    """
    Query a volume with voxelization.
    """
    (
        _,
        _,
        GaussianVoxelizationSettings,
        GaussianVoxelizer,
    ) = _load_raster_backend(getattr(pipe, "pose_enabled", False))

    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz
    density = pc.get_density

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    vol_pred, radii = voxelizer(
        means3D=means3D,
        opacities=density,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    pipe: PipelineParams,
    scaling_modifier=1.0,
):
    """
    Render an X-ray projection with rasterization.
    """

    # Evaluation/reporting code also calls render() directly. Refresh here so
    # attached calibration parameters are reflected in the matrices even when
    # the caller is not the main training loop.
    if (
        getattr(viewpoint_camera, "parallel_pose", None) is not None
        or getattr(viewpoint_camera, "cone_pose", None) is not None
    ):
        viewpoint_camera.refresh_parameters()

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    mode = viewpoint_camera.mode
    if mode == 0:
        if getattr(pipe, "pose_enabled", False):
            tanfovx = torch.tensor([1.0], device=pc.get_xyz.device)
            tanfovy = torch.tensor([1.0], device=pc.get_xyz.device)
        else:
            tanfovx = 1.0
            tanfovy = 1.0
    elif mode == 1:
        if getattr(pipe, "pose_enabled", False):
            tanfovx = torch.tan(
                torch.as_tensor(
                    viewpoint_camera.FoVx,
                    dtype=torch.float32,
                    device=pc.get_xyz.device,
                )
                * 0.5
            )
            tanfovy = torch.tan(
                torch.as_tensor(
                    viewpoint_camera.FoVy,
                    dtype=torch.float32,
                    device=pc.get_xyz.device,
                )
                * 0.5
            )
        else:
            tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
            tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    else:
        raise ValueError("Unsupported mode!")

    pose_backend = getattr(pipe, "pose_enabled", False)
    (
        GaussianRasterizationSettings,
        GaussianRasterizer,
        _,
        _,
    ) = _load_raster_backend(getattr(pipe, "pose_enabled", False))

    if pose_backend:
        # The target repository's differentiable backend receives camera
        # tensors as autograd inputs rather than storing them in the settings.
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            scale_modifier=scaling_modifier,
            prefiltered=False,
            mode=viewpoint_camera.mode,
            debug=pipe.debug,
        )
    else:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            mode=viewpoint_camera.mode,
            debug=pipe.debug,
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    density = pc.get_density

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if pose_backend:
        rendered_image, radii = rasterizer(
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            campos=viewpoint_camera.camera_center,
            means3D=means3D,
            means2D=means2D,
            opacities=density,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    else:
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=density,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
