"""Learnable cone-beam pose offsets compatible with the target repository."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ConePose(nn.Module):
    def __init__(self, n_views: int, cfg: Dict, device: str = "cuda"):
        super().__init__()
        dev = torch.device(device)
        self.register_buffer("device_ref", torch.zeros((), device=dev))
        self.optimize_translation = bool(cfg.get("optimize_translation", True))
        self.optimize_quaternion = bool(cfg.get("optimize_quaternion", True))
        self.raw_translation = (
            nn.Parameter(torch.zeros(n_views, 3, 1, device=dev))
            if self.optimize_translation
            else None
        )
        self.raw_quaternion = (
            nn.Parameter(torch.zeros(n_views, 4, device=dev))
            if self.optimize_quaternion
            else None
        )

    def values(self):
        return {
            "delta_translation": self.raw_translation,
            "delta_quaternion": self.raw_quaternion,
        }

    def regularization(self):
        values = self.values()
        loss = torch.zeros((), device=self.device_ref.device)
        if values["delta_translation"] is not None:
            loss = loss + values["delta_translation"].pow(2).mean()
        if values["delta_quaternion"] is not None:
            loss = loss + values["delta_quaternion"].pow(2).mean()
        return loss

    def optimizable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


class StaticConePose:
    """Non-trainable cone-beam pose container used during evaluation."""

    def __init__(self, values, device="cuda"):
        loaded = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in values.items()
        }
        self._values = {
            "delta_translation": loaded.get("delta_translation"),
            "delta_quaternion": loaded.get("delta_quaternion"),
        }

    @classmethod
    def from_npz(cls, path, device="cuda"):
        import numpy as np

        data = np.load(path)
        return cls({key: data[key] for key in data.files}, device=device)

    def values(self):
        return self._values

    def regularization(self):
        for value in self._values.values():
            if value is not None:
                return torch.zeros((), device=value.device)
        return torch.zeros(())
