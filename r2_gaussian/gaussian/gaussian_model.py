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
import sys
import torch
from torch import nn
import numpy as np
import pickle
from plyfile import PlyData, PlyElement

sys.path.append("./")

from simple_knn._C import distCUDA2
from r2_gaussian.utils.general_utils import t2a
from r2_gaussian.utils.system_utils import mkdir_p
from r2_gaussian.utils.gaussian_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    inverse_softplus,
    strip_symmetric,
    build_scaling_rotation,
)

EPS = 1e-5


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if self.scale_bound is not None:
            scale_min_bound, scale_max_bound = self.scale_bound
            assert (
                scale_min_bound < scale_max_bound
            ), "scale_min must be smaller than scale_max."
            self.scaling_activation = (
                lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound)
                + scale_min_bound
            )
            self.scaling_inverse_activation = lambda x: inverse_sigmoid(
                torch.relu((x - scale_min_bound) / (scale_max_bound - scale_min_bound))
            )
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.density_activation = torch.nn.Softplus()  # use softplus for [0, +inf]
        self.density_inverse_activation = inverse_softplus

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, scale_bound=None):
        self._xyz = torch.empty(0)  # world coordinate
        self._scaling = torch.empty(0)  # 3d scale
        self._rotation = torch.empty(0)  # rotation expressed in quaternions
        self._density = torch.empty(0)  # density
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        # Optional shape-ablation bookkeeping.  It is inactive unless the
        # training loop enables shape regularization.
        self.shape_bad_count = None
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.scale_bound = scale_bound
        self.setup_functions()

    def capture(self):
        captured = (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.scale_bound,
        )
        if self.shape_bad_count is not None:
            captured += (self.shape_bad_count,)
        return captured

    def restore(self, model_args, training_args):
        restored_shape_bad_count = None
        if len(model_args) >= 11:
            restored_shape_bad_count = model_args[10]
        (
            self._xyz,
            self._scaling,
            self._rotation,
            self._density,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.scale_bound,
        ) = model_args[:10]
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.setup_functions()  # Reset activation functions
        if restored_shape_bad_count is not None:
            self.shape_bad_count = restored_shape_bad_count.to(self._xyz.device)
        else:
            self.shape_bad_count = None

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_density(self):
        return self.density_activation(self._density)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def create_from_pcd(self, xyz, density, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        self.shape_bad_count = None

        fused_point_cloud = torch.tensor(xyz).float().cuda()
        print(
            "Initialize gaussians from {} estimated points".format(
                fused_point_cloud.shape[0]
            )
        )
        fused_density = (
            self.density_inverse_activation(torch.tensor(density)).float().cuda()
        )
        dist = torch.sqrt(
            torch.clamp_min(
                distCUDA2(fused_point_cloud),
                0.001**2,
            )
        )
        if self.scale_bound is not None:
            dist = torch.clamp(
                dist, self.scale_bound[0] + EPS, self.scale_bound[1] - EPS
            )  # Avoid overflow

        scales = self.scaling_inverse_activation(dist)[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._density = nn.Parameter(fused_density.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        #! Generate one gaussian for debugging purpose
        if False:
            print("Initialize one gaussian")
            fused_xyz = (
                torch.tensor([[0.0, 0.0, 0.0]]).float().cuda()
            )  # position: [0,0,0]
            fused_density = self.density_inverse_activation(
                torch.tensor([[0.8]]).float().cuda()
            )  # density: 0.8
            scales = self.scaling_inverse_activation(
                torch.tensor([[0.5, 0.5, 0.5]]).float().cuda()
            )  # scale: 0.5
            rots = (
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).float().cuda()
            )  # quaternion: [1, 0, 0, 0]
            # rots = torch.tensor([[0.966, -0.259, 0, 0]]).float().cuda()
            self._xyz = nn.Parameter(fused_xyz.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._density = nn.Parameter(fused_density.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.shape_bad_count is not None and self.shape_bad_count.shape[0] != self.get_xyz.shape[0]:
            self.shape_bad_count = None

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._density],
                "lr": training_args.density_lr_init * self.spatial_lr_scale,
                "name": "density",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr_init * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr_init * self.spatial_lr_scale,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            max_steps=training_args.position_lr_max_steps,
        )
        self.density_scheduler_args = get_expon_lr_func(
            lr_init=training_args.density_lr_init * self.spatial_lr_scale,
            lr_final=training_args.density_lr_final * self.spatial_lr_scale,
            max_steps=training_args.density_lr_max_steps,
        )
        self.scaling_scheduler_args = get_expon_lr_func(
            lr_init=training_args.scaling_lr_init * self.spatial_lr_scale,
            lr_final=training_args.scaling_lr_final * self.spatial_lr_scale,
            max_steps=training_args.scaling_lr_max_steps,
        )
        self.rotation_scheduler_args = get_expon_lr_func(
            lr_init=training_args.rotation_lr_init * self.spatial_lr_scale,
            lr_final=training_args.rotation_lr_final * self.spatial_lr_scale,
            max_steps=training_args.rotation_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "density":
                lr = self.density_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "scaling":
                lr = self.scaling_scheduler_args(iteration)
                param_group["lr"] = lr
            if param_group["name"] == "rotation":
                lr = self.rotation_scheduler_args(iteration)
                param_group["lr"] = lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        l.append("density")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        # We save pickle files to store more information

        mkdir_p(os.path.dirname(path))

        xyz = t2a(self._xyz)
        densities = t2a(self._density)
        scale = t2a(self._scaling)
        rotation = t2a(self._rotation)

        out = {
            "xyz": xyz,
            "density": densities,
            "scale": scale,
            "rotation": rotation,
            "scale_bound": self.scale_bound,
        }
        with open(path, "wb") as f:
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)

    def reset_density(self, reset_density=1.0):
        densities_new = self.density_inverse_activation(
            torch.min(
                self.get_density, torch.ones_like(self.get_density) * reset_density
            )
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
        self._density = optimizable_tensors["density"]

    def load_ply(self, path):
        # We load pickle file.
        with open(path, "rb") as f:
            data = pickle.load(f)

        self._xyz = nn.Parameter(
            torch.tensor(data["xyz"], dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._density = nn.Parameter(
            torch.tensor(
                data["density"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(
                data["scale"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(
                data["rotation"], dtype=torch.float, device="cuda"
            ).requires_grad_(True)
        )
        self.scale_bound = data["scale_bound"]
        self.shape_bad_count = None
        self.setup_functions()  # Reset activation functions

    def replace_tensor_to_optimizer(self, tensor, name, reset_mask=None):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    if reset_mask is None:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    else:
                        reset_mask = reset_mask.to(device=tensor.device, dtype=torch.bool)
                        state_mask = reset_mask.reshape(
                            (reset_mask.shape[0],) + (1,) * (tensor.ndim - 1)
                        )
                        stored_state["exp_avg"] = torch.where(
                            state_mask, torch.zeros_like(stored_state["exp_avg"]),
                            stored_state["exp_avg"],
                        )
                        stored_state["exp_avg_sq"] = torch.where(
                            state_mask, torch.zeros_like(stored_state["exp_avg_sq"]),
                            stored_state["exp_avg_sq"],
                        )

                    del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                if stored_state is not None:
                    self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.shape_bad_count is not None:
            self.shape_bad_count = self.shape_bad_count[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_densities,
        new_scaling,
        new_rotation,
        new_max_radii2D,
    ):
        d = {
            "xyz": new_xyz,
            "density": new_densities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat([self.max_radii2D, new_max_radii2D], dim=-1)
        if self.shape_bad_count is not None:
            self.shape_bad_count = torch.cat(
                [self.shape_bad_count,
                 torch.zeros(new_xyz.shape[0], device=new_xyz.device, dtype=torch.long)]
            )

    def enable_shape_tracking(self):
        """Initialize per-Gaussian persistent severe-shape counters."""
        self.shape_bad_count = torch.zeros(
            self.get_xyz.shape[0], device=self.get_xyz.device, dtype=torch.long
        )

    def shape_repair(self, mask, needle_threshold, sheet_threshold, max_density_gain=4.0,
                     needle_enabled=True, sheet_enabled=True):
        """Clamp severe axes while approximately preserving Gaussian mass."""
        if not mask.any():
            return 0
        scales = self.get_scaling.detach()
        repaired = scales.clone()
        order = torch.argsort(scales, dim=1)
        short = order[:, 0]
        middle = order[:, 1]
        long = order[:, 2]
        rows = torch.arange(scales.shape[0], device=scales.device)
        target_middle = scales[rows, short] * float(sheet_threshold)
        needle = (scales[rows, long] / scales[rows, middle]) > float(needle_threshold)
        sheet = (scales[rows, middle] / scales[rows, short]) > float(sheet_threshold)
        new_middle = torch.minimum(scales[rows, middle], target_middle)
        final_middle = new_middle if sheet_enabled else scales[rows, middle]
        new_long = torch.minimum(scales[rows, long], final_middle * float(needle_threshold))
        if needle_enabled:
            repaired[rows, long] = new_long
        if sheet_enabled:
            repaired[rows, middle] = torch.where(sheet, new_middle, scales[rows, middle])
        old_volume = torch.prod(scales, dim=1)
        new_volume = torch.prod(repaired, dim=1).clamp_min(1e-12)
        density = self.get_density.detach()
        density_new = density * (old_volume / new_volume)[:, None]
        density_new = torch.minimum(density_new, density * float(max_density_gain))
        scale_param = self.scaling_inverse_activation(repaired)
        density_param = density_new + torch.log(-torch.expm1(-density_new))
        valid = torch.isfinite(scale_param).all(dim=1) & torch.isfinite(density_param).all(dim=1)
        mask = mask & valid
        if not mask.any():
            return 0
        scale_param = torch.where(mask[:, None], scale_param, self._scaling.detach())
        density_param = torch.where(mask[:, None], density_param, self._density.detach())
        self._scaling = self.replace_tensor_to_optimizer(
            scale_param, "scaling", reset_mask=mask
        )["scaling"]
        self._density = self.replace_tensor_to_optimizer(
            density_param, "density", reset_mask=mask
        )["density"]
        self.shape_bad_count[mask] = 0
        return int(mask.sum().item())

    def shape_split(self, mask, children=2):
        """Split selected elongated Gaussians along their longest local axis."""
        if not mask.any() or children < 2:
            return 0
        selected = torch.nonzero(mask, as_tuple=False).flatten()
        scales = self.get_scaling.detach()[selected]
        rotations = self._rotation.detach()[selected]
        rot_mats = build_rotation(rotations)
        long_axis = torch.argmax(scales, dim=1)
        local_axis = torch.zeros((selected.shape[0], 3), device=scales.device)
        local_axis[torch.arange(selected.shape[0], device=scales.device), long_axis] = 1.0
        world_axis = torch.bmm(rot_mats, local_axis.unsqueeze(-1)).squeeze(-1)
        offsets = torch.linspace(-0.35, 0.35, children, device=scales.device)
        new_xyz = (self.get_xyz[selected, None, :] + world_axis[:, None, :] *
                   (scales[torch.arange(selected.shape[0], device=scales.device), long_axis][:, None, None] * offsets[None, :, None])).reshape(-1, 3)
        row_index = torch.arange(scales.shape[0], device=scales.device)
        axis_factor = torch.ones_like(scales)
        axis_factor[row_index, long_axis] = 1.0 / float(children)
        new_scales = (scales * axis_factor)[:, None, :].repeat(1, children, 1)
        new_scales = new_scales.reshape(-1, 3)
        scale_param = self.scaling_inverse_activation(new_scales)
        valid = torch.isfinite(scale_param).reshape(-1, children, 3).all(dim=2).all(dim=1)
        if not valid.all():
            safe_mask = torch.zeros_like(mask)
            safe_mask[selected[valid]] = True
            return self.shape_split(safe_mask, children) if safe_mask.any() else 0
        # Each child has roughly 1/children of the parent volume, so retaining
        # the parent's peak density approximately preserves summed Gaussian mass.
        new_density = self.get_density[selected].repeat_interleave(children, dim=0)
        new_rotation = rotations.repeat_interleave(children, dim=0)
        new_radii = self.max_radii2D[selected].repeat_interleave(children)
        self.densification_postfix(
            new_xyz,
            new_density + torch.log(-torch.expm1(-new_density)),
            scale_param,
            new_rotation,
            new_radii,
        )
        prune = torch.cat([mask, torch.zeros(new_xyz.shape[0], device=mask.device, dtype=torch.bool)])
        self.prune_points(prune)
        return int(selected.shape[0])

    def densify_and_split(self, grads, grad_threshold, densify_scale_threshold, N=2,
                          max_new_points=None):
        if isinstance(N, bool) or not isinstance(N, int) or N < 2:
            raise ValueError("Densification split N must be an integer >= 2")
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > densify_scale_threshold,
        )
        if max_new_points is not None:
            selected_pts_mask = self._limit_densification(
                selected_pts_mask, padded_grad, max_new_points // (N - 1))

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        child_scales = self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        if self.scale_bound is not None:
            # An anisotropic parent's short axes can cross the lower bound
            # even though its longest axis qualifies for splitting. Keep
            # children inside the sigmoid's representable open interval.
            lower, upper = self.scale_bound
            margin = (upper - lower) * torch.finfo(child_scales.dtype).eps
            child_scales = child_scales.clamp(lower + margin, upper - margin)
        new_scaling = self.scaling_inverse_activation(child_scales)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        # new_density = self._density[selected_pts_mask].repeat(N, 1)
        new_density = self.density_inverse_activation(
            self.get_density[selected_pts_mask].repeat(N, 1) * (1 / N)
        )
        new_max_radii2D = self.max_radii2D[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_density,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    @staticmethod
    def _limit_densification(mask, scores, limit):
        if (scores.ndim not in (1, 2) or (scores.ndim == 2 and scores.shape[1] != 1)
                or scores.shape[0] != mask.shape[0]):
            raise ValueError("Densification scores must contain one scalar per Gaussian: [N] or [N,1]")
        candidates = torch.nonzero(mask, as_tuple=False).flatten()
        if candidates.numel() <= limit:
            return mask
        result = torch.zeros_like(mask)
        if limit > 0:
            chosen = torch.topk(scores.reshape(-1)[candidates], int(limit)).indices
            result[candidates[chosen]] = True
        return result

    def densify_and_clone(self, grads, grad_threshold, densify_scale_threshold,
                          max_new_points=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= densify_scale_threshold,
        )
        if max_new_points is not None:
            selected_pts_mask = self._limit_densification(selected_pts_mask, grads, max_new_points)

        new_xyz = self._xyz[selected_pts_mask]
        # new_densities = self._density[selected_pts_mask]
        new_densities = self.density_inverse_activation(
            self.get_density[selected_pts_mask] * 0.5
        )
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_radii2D = self.max_radii2D[selected_pts_mask]

        self._density[selected_pts_mask] = new_densities

        self.densification_postfix(
            new_xyz,
            new_densities,
            new_scaling,
            new_rotation,
            new_max_radii2D,
        )

    def densify_and_prune(
        self,
        max_grad,
        min_density,
        max_screen_size,
        max_scale,
        max_num_gaussians,
        densify_scale_threshold,
        bbox=None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Densify Gaussians if Gaussians are fewer than threshold
        if densify_scale_threshold:
            if not max_num_gaussians or (
                max_num_gaussians and grads.shape[0] < max_num_gaussians
            ):
                remaining = lambda: (None if not max_num_gaussians else
                                     max(0, max_num_gaussians - self.get_xyz.shape[0]))
                self.densify_and_clone(grads, max_grad, densify_scale_threshold,
                                       max_new_points=remaining())
                self.densify_and_split(grads, max_grad, densify_scale_threshold,
                                       max_new_points=remaining())

        # Prune gaussians with too small density
        prune_mask = (self.get_density < min_density).squeeze()
        # Prune gaussians outside the bbox
        if bbox is not None:
            xyz = self.get_xyz
            prune_mask_xyz = (
                (xyz[:, 0] < bbox[0, 0])
                | (xyz[:, 0] > bbox[1, 0])
                | (xyz[:, 1] < bbox[0, 1])
                | (xyz[:, 1] > bbox[1, 1])
                | (xyz[:, 2] < bbox[0, 2])
                | (xyz[:, 2] > bbox[1, 2])
            )

            prune_mask = prune_mask | prune_mask_xyz

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
        if max_scale:
            big_points_ws = self.get_scaling.max(dim=1).values > max_scale
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

        return grads

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
