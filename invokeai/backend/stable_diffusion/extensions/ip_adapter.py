from __future__ import annotations

import math
import torch
from PIL.Image import Image
from contextlib import ExitStack
from typing import Any, List, Dict, Union, Optional
from diffusers import UNet2DConditionModel
from transformers import CLIPVisionModelWithProjection

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import IPAdapterConditioningInfo
from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
from .base import ExtensionBase, modifier
from ..denoise_context import DenoiseContext
from ..extensions_manager import ExtensionsManager
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import (
    CustomAttnProcessor2_0,
    IPAdapterAttentionWeights,
)


class IPAdapterExt(ExtensionBase):
    def __init__(
        self,
        node_context: "InvocationContext",
        exit_stack: ExitStack,
        model_id: "ModelIdentifierField",
        image_encoder_model_id: "ModelIdentifierField",
        images: List[Image],
        mask: torch.Tensor,
        target_blocks: List[str],
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
        priority: int,
    ):
        super().__init__(priority=priority)
        self.node_context = node_context
        self.exit_stack = exit_stack
        self.model_id = model_id
        self.image_encoder_model_id = image_encoder_model_id
        self.images = images
        self.mask = mask
        self.target_blocks = target_blocks
        self.weight = weight
        self.begin_step_percent = begin_step_percent
        self.end_step_percent = end_step_percent

        self.model: Optional[IPAdapter] = None
        self.conditioning: Optional[IPAdapterConditioningInfo] = None

    def patch_unet(self, state_dict: dict, unet: UNet2DConditionModel):
        for idx, name in enumerate(unet.attn_processors.keys()):
            if name.endswith("attn1.processor"):
                continue

            ip_adapter_weights = self.model.attn_weights.get_attention_processor_weights(idx)
            skip = True
            for block in self.target_blocks:
                if block in name:
                    skip = False
                    break

            assert isinstance(unet.attn_processors[name], CustomAttnProcessor2_0)
            unet.attn_processors[name].add_ip_adapter(
                IPAdapterAttentionWeights(
                    ip_adapter_weights=ip_adapter_weights,
                    skip=skip,
                )
            )

    def unpatch_unet(self, state_dict: dict, unet: UNet2DConditionModel):
        # nop, as it unpatched with attention processor
        pass

    @modifier("pre_unet_load")
    def preprocess_images(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        """Run the IPAdapter CLIPVisionModel, returning image prompt embeddings."""

        # HACK: save a bit of memory by not loading ip attention weights
        if True:
            with self.node_context.models.load(self.image_encoder_model_id) as image_encoder_model:
                ip_adapter_model = self.node_context.models.load(self.model_id).model
                assert isinstance(ip_adapter_model, IPAdapter)
                assert isinstance(image_encoder_model, CLIPVisionModelWithProjection)

                st_device = ip_adapter_model.device
                st_dtype = ip_adapter_model.dtype
                try:
                    ip_adapter_model.device = image_encoder_model.device
                    ip_adapter_model.dtype = image_encoder_model.dtype
                    ip_adapter_model._image_proj_model.to(device=image_encoder_model.device, dtype=image_encoder_model.dtype)

                    # Get image embeddings from CLIP and ImageProjModel.
                    positive_img_prompt_embeds, negative_img_prompt_embeds = ip_adapter_model.get_image_embeds(
                        self.images, image_encoder_model
                    )
                finally:
                    ip_adapter_model.device = st_device
                    ip_adapter_model.dtype = st_dtype
                    ip_adapter_model._image_proj_model.to(device=st_device, dtype=st_dtype)
        else:
            with (
                self.node_context.models.load(self.model_id) as ip_adapter_model,
                self.node_context.models.load(self.image_encoder_model_id) as image_encoder_model,
            ):
                assert isinstance(ip_adapter_model, IPAdapter)
                assert isinstance(image_encoder_model, CLIPVisionModelWithProjection)
                # Get image embeddings from CLIP and ImageProjModel.
                positive_img_prompt_embeds, negative_img_prompt_embeds = ip_adapter_model.get_image_embeds(
                    self.images, image_encoder_model
                )

        self.conditioning = IPAdapterConditioningInfo(positive_img_prompt_embeds, negative_img_prompt_embeds)


        _, _, latent_height, latent_width = ctx.latents.shape
        self.mask = self._preprocess_regional_prompt_mask(self.mask, latent_height, latent_width, dtype=ctx.latents.dtype)
        
        # HACK: save a bit memory by not loading ImageProjModel
        if True:
            self.model = self.node_context.models.load(self.model_id).model
            # move back to storage device on __exit__
            self.exit_stack.callback(self.model.attn_weights.to, device=self.model.device, dtype=self.model.dtype)
            # load attn weights to device
            self.model.attn_weights.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

        else:
            self.model = self.exit_stack.enter_context(self.node_context.models.load(self.model_id))
        
    @staticmethod
    def _preprocess_regional_prompt_mask(
        mask: Optional[torch.Tensor], target_height: int, target_width: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Preprocess a regional prompt mask to match the target height and width.
        If mask is None, returns a mask of all ones with the target height and width.
        If mask is not None, resizes the mask to the target height and width using 'nearest' interpolation.

        Returns:
            torch.Tensor: The processed mask. shape: (1, 1, target_height, target_width).
        """

        if mask is None:
            return torch.ones((1, 1, target_height, target_width), dtype=dtype)

        from invokeai.backend.util.mask import to_standard_float_mask
        mask = to_standard_float_mask(mask, out_dtype=dtype)

        tf = torchvision.transforms.Resize(
            (target_height, target_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # Add a batch dimension to the mask, because torchvision expects shape (batch, channels, h, w).
        mask = mask.unsqueeze(0)  # Shape: (1, h, w) -> (1, 1, h, w)
        resized_mask = tf(mask)
        return resized_mask

    @modifier("pre_unet_forward")
    def pre_unet_step(self, ctx: DenoiseContext):
        # skip if model not active in current step
        total_steps = len(ctx.timesteps)
        first_step = math.floor(self.begin_step_percent * total_steps)
        last_step = math.ceil(self.end_step_percent * total_steps)
        if ctx.step_index < first_step or ctx.step_index > last_step:
            return

        weight = self.weight
        if isinstance(weight, List):
            weight = weight[ctx.step_index]

        if ctx.conditioning_mode == "both":
            embeds = torch.stack([self.conditioning.uncond_image_prompt_embeds, self.conditioning.cond_image_prompt_embeds])
        elif ctx.conditioning_mode == "negative":
            embeds = torch.stack([self.conditioning.uncond_image_prompt_embeds])
        else: # elif ctx.conditioning_mode == "positive":
            embeds = torch.stack([self.conditioning.cond_image_prompt_embeds])

        if ctx.unet_kwargs.cross_attention_kwargs is None:
            ctx.unet_kwargs.cross_attention_kwargs = dict()

        regional_ip_data = ctx.unet_kwargs.cross_attention_kwargs.get("regional_ip_data", None)
        if regional_ip_data is None:
            regional_ip_data = RegionalIPData(
                image_prompt_embeds=[],
                scales=[],
                masks=[],
                dtype=ctx.latent_model_input.dtype,
                device=ctx.latent_model_input.device,
            )
            ctx.unet_kwargs.cross_attention_kwargs.update(dict(
                regional_ip_data=regional_ip_data,
            ))


        regional_ip_data.add(
            embeds=embeds,
            scale=weight,
            mask=self.mask,
        )
