from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, Union

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput

if TYPE_CHECKING:
    from invokeai.backend.flux.model import Flux
    from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
    from invokeai.backend.flux.extensions.xlabs_controlnet_extension import XLabsControlNetExtension
    from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
    from invokeai.backend.flux.extensions.xlabs_ip_adapter_extension import XLabsIPAdapterExtension
    from invokeai.backend.flux.extensions.dype_extension import DyPEExtension
    from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
    from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
    from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode, TextConditioningData


# @dataclass
# class ModelKwargs:
#     sample: torch.Tensor
#     timestep: Union[torch.Tensor, float, int]
#     encoder_hidden_states: torch.Tensor

#     class_labels: Optional[torch.Tensor] = None
#     timestep_cond: Optional[torch.Tensor] = None
#     attention_mask: Optional[torch.Tensor] = None
#     cross_attention_kwargs: Optional[Dict[str, Any]] = None
#     added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None
#     down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None
#     mid_block_additional_residual: Optional[torch.Tensor] = None
#     down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None
#     encoder_attention_mask: Optional[torch.Tensor] = None
#     # return_dict: bool = True


@dataclass
class DenoiseContext:
    """Context with all variables in denoise"""

    model: Flux

    # model input
    img: torch.Tensor
    img_ids: torch.Tensor
    pos_regional_prompting_extension: RegionalPromptingExtension
    neg_regional_prompting_extension: RegionalPromptingExtension | None

    # sampling parameters
    timesteps: list[float]
    scheduler: SchedulerMixin
    step_callback: Callable[[PipelineIntermediateState], None]
    guidance: float
    cfg_scale: list[float]
    inpaint_extension: RectifiedFlowInpaintExtension | None
    controlnet_extensions: list[XLabsControlNetExtension | InstantXControlNetExtension]
    pos_ip_adapter_extensions: list[XLabsIPAdapterExtension]
    neg_ip_adapter_extensions: list[XLabsIPAdapterExtension]

    # extra img tokens (channel-wise)
    img_cond: torch.Tensor | None

    # extra img tokens (sequence-wise) - for Kontext conditioning
    img_cond_seq: torch.Tensor | None = None
    img_cond_seq_ids: torch.Tensor | None = None

    # DyPE extension for high-resolution generation
    dype_extension: DyPEExtension | None = None

    # local vars
    step_index: int | None = None
    user_step: int | None = None
    total_steps: int | None = None
    t_curr: float | None = None
    conditioning_mode: ConditioningMode | None = None

    # Dictionary for extensions to pass extra info about denoise process to other extensions.
    extra: dict = field(default_factory=dict)
