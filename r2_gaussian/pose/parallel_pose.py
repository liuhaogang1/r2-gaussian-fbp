"""Learnable, bounded geometry corrections for parallel-beam CT.

The module is deliberately independent from the Gaussian model.  It is only
constructed when pose calibration is enabled, so the default training path is
unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

PARALLEL_POSE_FORMAT_VERSION = 2


class ParallelPose(nn.Module):
    """Global and per-view corrections for a parallel-beam scan.

    Internal angles are radians and detector shifts are stored in detector
    pixels.  The camera converts detector pixels to the normalized scene
    coordinate system when it refreshes its matrices.
    """

    def __init__(self, n_views: int, cfg: Dict, device: str = "cuda"):
        super().__init__()
        self.n_views = int(n_views)
        self.device_name = device

        self.optimize_angle = bool(cfg.get("optimize_angle", False))
        self.optimize_global_shift = bool(
            cfg.get("optimize_global_detector_shift", False)
        )
        self.optimize_per_view_shift = bool(
            cfg.get("optimize_per_view_detector_shift", False)
        )
        self.optimize_beam_tilt = bool(cfg.get("optimize_beam_tilt", False))
        self.optimize_detector_roll = bool(
            cfg.get("optimize_detector_roll", False)
        )

        dev = torch.device(device)
        self.register_buffer(
            "max_angle",
            torch.tensor(float(cfg.get("angle_max_deg", 0.5)), device=dev)
            * torch.pi
            / 180.0,
        )
        self.register_buffer(
            "max_detector_shift",
            torch.tensor(float(cfg.get("detector_shift_max_pixel", 3.0)), device=dev),
        )
        self.register_buffer(
            "max_beam_tilt",
            torch.tensor(float(cfg.get("beam_tilt_max_deg", 0.5)), device=dev)
            * torch.pi
            / 180.0,
        )
        self.register_buffer(
            "max_detector_roll",
            torch.tensor(float(cfg.get("detector_roll_max_deg", 0.5)), device=dev)
            * torch.pi
            / 180.0,
        )

        self.raw_angle = (
            nn.Parameter(torch.zeros(self.n_views, device=dev))
            if self.optimize_angle
            else None
        )
        self.raw_global_shift = (
            nn.Parameter(torch.zeros(2, device=dev))
            if self.optimize_global_shift
            else None
        )
        self.raw_view_shift = (
            nn.Parameter(torch.zeros(self.n_views, 2, device=dev))
            if self.optimize_per_view_shift
            else None
        )
        self.raw_beam_tilt = (
            nn.Parameter(torch.zeros(2, device=dev))
            if self.optimize_beam_tilt
            else None
        )
        self.raw_detector_roll = (
            nn.Parameter(torch.zeros(1, device=dev))
            if self.optimize_detector_roll
            else None
        )

    @staticmethod
    def _bounded(raw: Optional[torch.Tensor], maximum: torch.Tensor):
        return None if raw is None else maximum * torch.tanh(raw)

    @staticmethod
    def _center_and_bound(value: torch.Tensor, maximum: torch.Tensor):
        """Keep a per-view correction zero-mean without exceeding its limit."""
        centered = value - torch.mean(value, dim=0, keepdim=True)
        max_abs = torch.amax(torch.abs(centered), dim=0, keepdim=True)
        scale = torch.clamp(
            maximum / torch.clamp(max_abs, min=1.0e-12),
            max=1.0,
        )
        return centered * scale

    def values(self) -> Dict[str, Optional[torch.Tensor]]:
        delta_angle = self._bounded(self.raw_angle, self.max_angle)
        if delta_angle is not None:
            # Fix the global angular gauge: only relative view errors are learned.
            delta_angle = self._center_and_bound(delta_angle, self.max_angle)

        view_shift = self._bounded(
            self.raw_view_shift, self.max_detector_shift
        )
        if view_shift is not None:
            # The global detector shift carries the mean component.
            view_shift = self._center_and_bound(
                view_shift, self.max_detector_shift
            )

        return {
            "delta_angle": delta_angle,
            "global_shift": self._bounded(
                self.raw_global_shift, self.max_detector_shift
            ),
            "view_shift": view_shift,
            "beam_tilt": self._bounded(
                self.raw_beam_tilt, self.max_beam_tilt
            ),
            "detector_roll": self._bounded(
                self.raw_detector_roll, self.max_detector_roll
            ),
        }

    def regularization(self, values=None) -> torch.Tensor:
        values = self.values() if values is None else values
        loss = torch.zeros((), device=self.max_angle.device)
        if values["delta_angle"] is not None and self.n_views > 1:
            loss = loss + (
                values["delta_angle"][1:] - values["delta_angle"][:-1]
            ).pow(2).mean()
        if values["view_shift"] is not None and self.n_views > 1:
            loss = loss + (
                values["view_shift"][1:] - values["view_shift"][:-1]
            ).pow(2).mean()
        return loss

    def optimizable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class StaticParallelPose:
    """Non-trainable pose container used by test.py for saved offsets."""

    def __init__(self, values, device="cuda"):
        loaded = {
            key: torch.as_tensor(value, device=device)
            for key, value in values.items()
        }
        self._values = {
            "delta_angle": loaded.get("delta_angle"),
            "global_shift": loaded.get("global_shift"),
            "view_shift": loaded.get("view_shift"),
            "beam_tilt": loaded.get("beam_tilt"),
            "detector_roll": loaded.get("detector_roll"),
        }

    @classmethod
    def from_npz(cls, path, device="cuda"):
        import numpy as np

        with np.load(path) as data:
            version = (
                int(data["pose_format_version"].item())
                if "pose_format_version" in data
                else None
            )
            if version != PARALLEL_POSE_FORMAT_VERSION:
                raise ValueError(
                    "This parallel pose file uses the legacy transposed-rotation "
                    "convention (or has no format version). Retrain with the "
                    "corrected geometry before loading it."
                )
            values = {
                key: data[key]
                for key in (
                    "delta_angle",
                    "global_shift",
                    "view_shift",
                    "beam_tilt",
                    "detector_roll",
                )
                if key in data
            }
        return cls(values, device=device)

    def values(self):
        return self._values

    def regularization(self):
        for value in self._values.values():
            if value is not None:
                return torch.zeros((), device=value.device)
        return torch.zeros(())
