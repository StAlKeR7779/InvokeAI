from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from pydantic import Field

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import TextConditioningData
from .base import AddonBase

@dataclass
class DepthModelAddon(AddonBase):
    depth_mask: torch.Tensor

    def pre_unet_step(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        step_index: int,
        total_steps: int,
        conditioning_data: TextConditioningData,

        unet_kwargs: Dict[str, Any],
        conditioning_mode: str,
    ):
        batch_size = sample.shape[0]
        if conditioning_mode == "both":
            batch_size *= 2

        b_depth_mask = torch.cat([self.depth_mask] * batch_size).to(device=sample.device, dtype=sample.dtype)

        unet_kwargs.update(dict(
            extra_channels=b_depth_mask,
        ))
