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
from r2_gaussian.utils.geometry_utils import parallel_detector_half_extents


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


def query_points(
    pc: GaussianModel,
    points_world: torch.Tensor,
    pipe: PipelineParams,
    chunk_size: int = 1024,
    gaussian_chunk_size: int = 8192,
):
    """Differentiably evaluate the Gaussian density at sparse world points.

    This is the sparse counterpart of :func:`query`.  It intentionally uses
    the same unnormalised Gaussian sum as the CUDA voxelizer, while avoiding
    allocation of a dense query volume.  The points are split into chunks to
    keep the ``points x gaussians`` temporary bounded.
    """
    if points_world.ndim != 2 or points_world.shape[-1] != 3:
        raise ValueError("points_world must have shape [M, 3]")
    if points_world.numel() == 0:
        return points_world.new_empty((0,))
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if gaussian_chunk_size < 1:
        raise ValueError("gaussian_chunk_size must be positive")

    points_world = points_world.to(device=pc.get_xyz.device, dtype=pc.get_xyz.dtype)
    means = pc.get_xyz
    density = pc.get_density.reshape(-1)
    # get_covariance returns [xx, xy, xz, yy, yz, zz], matching the CUDA
    # voxelizer's symmetric covariance layout. Add a tiny diagonal floor to
    # avoid rare singular matrices after densification.
    cov6 = pc.get_covariance(1.0)
    cov = torch.zeros((cov6.shape[0], 3, 3), device=means.device, dtype=means.dtype)
    cov[:, 0, 0] = cov6[:, 0]
    cov[:, 0, 1] = cov6[:, 1]
    cov[:, 1, 0] = cov6[:, 1]
    cov[:, 0, 2] = cov6[:, 2]
    cov[:, 2, 0] = cov6[:, 2]
    cov[:, 1, 1] = cov6[:, 3]
    cov[:, 1, 2] = cov6[:, 4]
    cov[:, 2, 1] = cov6[:, 4]
    cov[:, 2, 2] = cov6[:, 5]
    cov = cov + torch.eye(3, device=means.device, dtype=means.dtype)[None] * 1e-10
    precision = torch.linalg.inv(cov)

    outputs = []
    # The Gaussian count is intentionally kept in one dimension here. This
    # preserves gradients to xyz, density, scale and rotation parameters.
    for start in range(0, points_world.shape[0], chunk_size):
        points = points_world[start : start + chunk_size]
        values = points.new_zeros((points.shape[0],))
        for g_start in range(0, means.shape[0], gaussian_chunk_size):
            g_end = g_start + gaussian_chunk_size
            delta = points[:, None, :] - means[None, g_start:g_end, :]
            mahal = torch.einsum(
                "mni,nij,mnj->mn",
                delta,
                precision[g_start:g_end],
                delta,
            )
            values = values + torch.exp(-0.5 * mahal).matmul(
                density[g_start:g_end]
            )
        outputs.append(values)
    return torch.cat(outputs, dim=0)


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
        # The CUDA covariance Jacobian uses width/(2*tanfovx) and
        # height/(2*tanfovy). For parallel beams these inputs are the actual
        # scene-normalized detector half extents, rather than angular tangents.
        half_u, half_v = parallel_detector_half_extents(viewpoint_camera.scanner_cfg)
        if getattr(pipe, "pose_enabled", False):
            tanfovx = torch.tensor([half_u], dtype=torch.float32, device=pc.get_xyz.device)
            tanfovy = torch.tensor([half_v], dtype=torch.float32, device=pc.get_xyz.device)
        else:
            tanfovx = half_u
            tanfovy = half_v
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
