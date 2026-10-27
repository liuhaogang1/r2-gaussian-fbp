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

import torch
import math
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from r2_gaussian.utils.geometry_utils import parallel_projection_matrix


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    try:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    except:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)  # w2c
    return np.float32(Rt)


def getProjectionMatrix(fovX, fovY, mode, scanner_cfg):
    if mode == 0:  # Parallel beam
        P = torch.as_tensor(parallel_projection_matrix(scanner_cfg))
    elif mode == 1:
        znear = 0.01
        zfar = 100.0
        fovY = torch.as_tensor(fovY, dtype=torch.float32)
        fovX = torch.as_tensor(fovX, dtype=torch.float32)
        tanHalfFovY = torch.tan(fovY / 2)
        tanHalfFovX = torch.tan(fovX / 2)

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
    else:
        raise ValueError("Unsupported mode!")
    return P


def getParallelProjectionMatrix(detector_shift, scanner_cfg, device="cuda"):
    """Build the affine projection used by the parallel-beam rasterizer.

    ``detector_shift`` is in the normalized scene unit used by ``scanner_cfg``
    (the dataset reader scales the physical scanner geometry before cameras
    are constructed).  The sign convention is the same as the rasterizer's
    normalized detector coordinates.
    """
    p = torch.as_tensor(
        parallel_projection_matrix(scanner_cfg), dtype=torch.float32, device=device
    )
    shift_u = detector_shift[0]
    shift_v = detector_shift[1]
    detector_u = torch.as_tensor(
        float(scanner_cfg["sDetector"][1]), device=device
    )
    detector_v = torch.as_tensor(
        float(scanner_cfg["sDetector"][0]), device=device
    )
    p[0, 3] = 2.0 * shift_u / detector_u
    p[1, 3] = 2.0 * shift_v / detector_v
    return p


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))
