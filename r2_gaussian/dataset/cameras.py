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
from torch import nn
import numpy as np

sys.path.append("./")
from r2_gaussian.utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrix,
    getParallelProjectionMatrix,
)


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        scanner_cfg,
        R,
        T,
        angle,
        mode,
        FoVx,
        FoVy,
        image,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.angle = angle
        self.scanner_cfg = scanner_cfg
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.mode = mode
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                fovX=self.FoVx,
                fovY=self.FoVy,
                mode=mode,
                scanner_cfg=scanner_cfg,
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # These are populated only when optional pose calibration is enabled.
        self.parallel_pose = None
        self.cone_pose = None
        self.pose_index = None
        self._base_world_view_transform = self.world_view_transform.detach().clone()
        self._base_projection_matrix = self.projection_matrix.detach().clone()

    def attach_parallel_pose(self, pose_model, pose_index):
        if self.mode != 0:
            raise ValueError("Parallel pose calibration requires mode='parallel'.")
        self.parallel_pose = pose_model
        self.pose_index = int(pose_index)

    def attach_cone_pose(self, pose_model, pose_index):
        if self.mode != 1:
            raise ValueError("Cone pose calibration requires mode='cone'.")
        self.cone_pose = pose_model
        self.pose_index = int(pose_index)

    @staticmethod
    def _rot_x(angle, device):
        c, s = torch.cos(angle), torch.sin(angle)
        r = torch.eye(4, device=device)
        r[1, 1], r[1, 2] = c, -s
        r[2, 1], r[2, 2] = s, c
        return r

    @staticmethod
    def _rot_y(angle, device):
        c, s = torch.cos(angle), torch.sin(angle)
        r = torch.eye(4, device=device)
        r[0, 0], r[0, 2] = c, s
        r[2, 0], r[2, 2] = -s, c
        return r

    @staticmethod
    def _rot_z(angle, device):
        c, s = torch.cos(angle), torch.sin(angle)
        r = torch.eye(4, device=device)
        r[0, 0], r[0, 1] = c, -s
        r[1, 0], r[1, 1] = s, c
        return r

    def refresh_parameters(self):
        """Refresh differentiable camera matrices for parallel calibration."""
        if self.cone_pose is not None:
            values = self.cone_pose.values()
            device = self._base_world_view_transform.device
            init_translation = torch.as_tensor(
                self.T, dtype=torch.float32, device=device
            ).view(3, 1)
            init_rotation = torch.as_tensor(
                self.R, dtype=torch.float32, device=device
            ).t()
            init_quaternion = self._rotation_matrix_to_quaternion(init_rotation)

            translation = init_translation
            if values["delta_translation"] is not None:
                translation = (
                    translation
                    + values["delta_translation"][self.pose_index]
                )

            quaternion = init_quaternion
            if values["delta_quaternion"] is not None:
                quaternion = (
                    quaternion
                    + values["delta_quaternion"][self.pose_index]
                )
            rotation = self._quaternion_to_rotation_matrix(quaternion)

            rt = torch.cat((rotation, translation), dim=1)
            last_row = torch.tensor(
                [[0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )
            self.world_view_transform = torch.cat((rt, last_row), dim=0).t()
            self.projection_matrix = (
                getProjectionMatrix(
                    fovX=self.FoVx,
                    fovY=self.FoVy,
                    mode=self.mode,
                    scanner_cfg=self.scanner_cfg,
                )
                .transpose(0, 1)
                .to(device)
            )
            self.full_proj_transform = (
                self.world_view_transform.unsqueeze(0)
                .bmm(self.projection_matrix.unsqueeze(0))
                .squeeze(0)
            )
            self.camera_center = self.world_view_transform.inverse()[3, :3]
            return

        if self.parallel_pose is None:
            return

        values = self.parallel_pose.values()
        device = self._base_world_view_transform.device
        base_c2w = torch.linalg.inv(self._base_world_view_transform)
        corrected_c2w = base_c2w

        delta_angle = values["delta_angle"]
        if delta_angle is not None:
            corrected_c2w = self._rot_z(
                delta_angle[self.pose_index], device
            ) @ corrected_c2w

        beam_tilt = values["beam_tilt"]
        if beam_tilt is not None:
            corrected_c2w = (
                self._rot_y(beam_tilt[1], device)
                @ self._rot_x(beam_tilt[0], device)
                @ corrected_c2w
            )

        detector_roll = values["detector_roll"]
        if detector_roll is not None:
            # Roll is defined in the camera-local frame.
            corrected_c2w = corrected_c2w @ self._rot_z(
                detector_roll[0], device
            )

        self.world_view_transform = torch.linalg.inv(corrected_c2w)

        shift_pixels = torch.zeros(2, device=device)
        if values["global_shift"] is not None:
            shift_pixels = shift_pixels + values["global_shift"]
        if values["view_shift"] is not None:
            shift_pixels = shift_pixels + values["view_shift"][self.pose_index]

        d_detector = torch.as_tensor(
            self.scanner_cfg["dDetector"],
            dtype=torch.float32,
            device=device,
        )
        # Detector configuration is [v, u].
        shift_scene = torch.stack(
            [shift_pixels[0] * d_detector[1], shift_pixels[1] * d_detector[0]]
        )
        self.projection_matrix = getParallelProjectionMatrix(
            shift_scene, self.scanner_cfg, device=device
        ).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def _rotation_matrix_to_quaternion(matrix):
        # Quaternion layout matches the target repository: [w, x, y, z].
        trace = matrix.trace()
        w = torch.sqrt(torch.clamp(trace + 1.0, min=1.0e-8)) / 2.0
        denom = 4.0 * w + 1.0e-8
        x = (matrix[2, 1] - matrix[1, 2]) / denom
        y = (matrix[0, 2] - matrix[2, 0]) / denom
        z = (matrix[1, 0] - matrix[0, 1]) / denom
        return torch.stack((w, x, y, z))

    @staticmethod
    def _quaternion_to_rotation_matrix(quaternion):
        q = torch.nn.functional.normalize(quaternion, dim=0)
        w, x, y, z = q
        return torch.stack(
            (
                torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w))),
                torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w))),
                torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
            )
        )


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
