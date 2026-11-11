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
import math
import os.path as osp
from argparse import ArgumentParser, Namespace

sys.path.append("./")
from r2_gaussian.utils.argument_utils import ParamGroup


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"
        self.ply_path = ""  # Path to initialization point cloud (if None, we will try to find `init_*.npy`.)
        self.scale_min = 0.0005  # percent of volume size
        self.scale_max = 0.5  # percent of volume size
        self.eval = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = osp.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.compute_cov3D_python = False
        self.debug = False
        self.projection_data_range = 0.0  # 0 selects the evaluation GT global maximum.
        super().__init__(parser, "Pipeline Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        for key, value in vars(self).items():
            if getattr(group, key, None) is None:
                setattr(group, key, value)
        if not math.isfinite(group.projection_data_range) or group.projection_data_range < 0:
            raise ValueError("projection_data_range must be 0 (automatic) or positive and finite")
        return group


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0002
        self.position_lr_final = 0.00002
        self.position_lr_max_steps = 30_000
        self.density_lr_init = 0.01
        self.density_lr_final = 0.001
        self.density_lr_max_steps = 30_000
        self.scaling_lr_init = 0.005
        self.scaling_lr_final = 0.0005
        self.scaling_lr_max_steps = 30_000
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.0001
        self.rotation_lr_max_steps = 30_000
        self.lambda_dssim = 0.25
        self.lambda_tv = 0.05
        self.tv_vol_size = 32
        self.density_min_threshold = 0.00001
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 5.0e-5
        self.densify_scale_threshold = 0.1  # percent of volume size
        self.max_screen_size = None
        self.max_scale = None  # percent of volume size
        self.max_num_gaussians = 500_000

        # Optional anomalous Gaussian-shape suppression. Disabled by default.
        self.shape_enabled = False
        self.shape_needle_enabled = False
        self.shape_sheet_enabled = False
        self.shape_needle_threshold = 10.0
        self.shape_sheet_threshold = 10.0
        self.lambda_shape_needle = 0.0
        self.lambda_shape_sheet = 0.0
        self.shape_start_iter = 500
        self.shape_end_iter = 15000
        self.shape_ramp_iters = 1000
        self.shape_check_interval = 100
        self.shape_patience = 3
        self.shape_hard_needle_ratio = 50.0
        self.shape_hard_sheet_ratio = 50.0
        self.shape_hard_action = "none"
        self.shape_low_density_threshold = 0.0
        self.shape_low_mass_threshold = 0.0
        self.shape_repair_max_density_gain = 4.0
        self.shape_split_count = 2
        self.shape_max_hard_actions_per_check = 32

        # Optional sparse initialization-volume slice supervision.
        self.slice_loss_enabled = False
        self.slice_reference_path = ""
        self.slice_axis = "z"
        self.slice_start_iter = 500
        self.slice_end_iter = 15000
        self.slice_interval = 50
        self.slice_filter_threshold = 0.01
        self.slice_filter_ratio = 10.0
        self.slice_max_points = 200_000
        self.slice_query_chunk_size = 4096
        self.lambda_slice = 0.05
        self.slice_seed = 0

        # Optional pose-calibration branch.  All switches are disabled by
        # default so the historical training path remains unchanged.
        self.pose_enabled = False
        self.pose_mode = "auto"
        self.parallel_optimize_angle = False
        self.parallel_optimize_global_detector_shift = False
        self.parallel_optimize_per_view_detector_shift = False
        self.parallel_optimize_beam_tilt = False
        self.parallel_optimize_detector_roll = False
        self.parallel_angle_max_deg = 0.5
        self.parallel_detector_shift_max_pixel = 3.0
        self.parallel_beam_tilt_max_deg = 0.5
        self.parallel_detector_roll_max_deg = 0.5
        self.pose_lr = 0.0001
        self.pose_lambda_smooth = 0.1
        self.cone_optimize_translation = True
        self.cone_optimize_quaternion = True
        self.pose_lambda_magnitude = 0.0
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_string)
    from r2_gaussian.utils.config_io import load_saved_config
    model_path = getattr(args_cmdline, "model_path", None)
    merged_dict = load_saved_config(model_path) if model_path else {}
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
