from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
import torchvision
from diffusers import UNet2DConditionModel
from PIL.Image import Image
from transformers import CLIPVisionModelWithProjection

from invokeai.backend.ip_adapter.ip_adapter import IPAdapter
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode, IPAdapterConditioningInfo
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import (
    CustomAttnProcessor2_0,
    IPAdapterAttentionWeights,
)
from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback
from invokeai.backend.util.mask import to_standard_float_mask
from invokeai.backend.stable_diffusion.extensions_manager import CallbackApi

if TYPE_CHECKING:
    from invokeai.app.invocations.model import ModelIdentifierField
    from invokeai.app.services.shared.invocation_context import InvocationContext
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager


class IPAdapterExt(ExtensionBase):
    def __init__(
        self,
        node_context: InvocationContext,
        exit_stack: ExitStack,
        model_id: ModelIdentifierField,
        image_encoder_model_id: ModelIdentifierField,
        images: List[Image],
        mask: torch.Tensor,
        target_blocks: List[str],
        weight: Union[float, List[float]],
        begin_step_percent: float,
        end_step_percent: float,
    ):
        super().__init__()
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

    @callback(CallbackApi.setup)
    def preprocess_images(self, ctx: DenoiseContext):
        """Run the IPAdapter CLIPVisionModel, returning image prompt embeddings."""

        # HACK: save a bit of memory by not loading ip attention weights on image processing
        # and by loading only attention weight on denoising
        if True:
            with self.node_context.models.load(self.image_encoder_model_id) as image_encoder_model:
                self.model = self.node_context.models.load(self.model_id).model
                assert isinstance(self.model, IPAdapter)
                assert isinstance(image_encoder_model, CLIPVisionModelWithProjection)

                st_device = self.model.device
                st_dtype = self.model.dtype

                def _move_ip_adapter_to_storage_device(model):
                    model.device = st_device
                    model.dtype = st_dtype
                    model._image_proj_model.to(device=st_device, dtype=st_dtype)
                    model.attn_weights.to(device=st_device, dtype=st_dtype)
                
                try:
                    self.model.device = image_encoder_model.device
                    self.model.dtype = image_encoder_model.dtype

                    # Get image embeddings from CLIP(image_encoder_model) and ImageProjModel(_image_proj_model).
                    self.model._image_proj_model.to(device=image_encoder_model.device, dtype=image_encoder_model.dtype)
                    positive_img_prompt_embeds, negative_img_prompt_embeds = self.model.get_image_embeds(
                        self.images, image_encoder_model
                    )
                    self.model._image_proj_model.to(device=st_device, dtype=st_dtype)

                    # load attention weights to device
                    self.model.attn_weights.to(device=ctx.inputs.orig_latents.device, dtype=ctx.inputs.orig_latents.dtype)
                    # move back to storage device on __exit__
                    self.exit_stack.callback(_move_ip_adapter_to_storage_device, self.model)
                except:
                    _move_ip_adapter_to_storage_device(self.model)
                    raise

        else:
            self.model = self.exit_stack.enter_context(self.node_context.models.load(self.model_id))
            with self.node_context.models.load(self.image_encoder_model_id) as image_encoder_model:
                assert isinstance(self.model, IPAdapter)
                assert isinstance(image_encoder_model, CLIPVisionModelWithProjection)
                # Get image embeddings from CLIP and ImageProjModel.
                positive_img_prompt_embeds, negative_img_prompt_embeds = self.model.get_image_embeds(
                    self.images, image_encoder_model
                )

        self.conditioning = IPAdapterConditioningInfo(positive_img_prompt_embeds, negative_img_prompt_embeds)

        _, _, latent_height, latent_width = ctx.inputs.orig_latents.shape
        tmp = self._preprocess_regional_prompt_mask(
            self.mask, latent_height, latent_width, dtype=ctx.inputs.orig_latents.dtype
        )
        self.mask_tensor = self._prepare_masks(
            tmp,
            max_downscale_factor=8,
            device=ctx.inputs.orig_latents.device,
            dtype=ctx.inputs.orig_latents.dtype,
        )

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

        mask = to_standard_float_mask(mask, out_dtype=dtype)

        tf = torchvision.transforms.Resize(
            (target_height, target_width), interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # Add a batch dimension to the mask, because torchvision expects shape (batch, channels, h, w).
        mask = mask.unsqueeze(0)  # Shape: (1, h, w) -> (1, 1, h, w)
        resized_mask = tf(mask)
        return resized_mask

    def _prepare_masks(
        self, mask_tensor: torch.Tensor, max_downscale_factor: int, device: torch.device, dtype: torch.dtype
    ) -> dict[int, torch.Tensor]:
        mask_tensor = mask_tensor.to(device=device, dtype=dtype, copy=True)
        masks_by_seq_len: dict[int, torch.Tensor] = {}

        # Downsample the spatial dimensions by factors of 2 until max_downscale_factor is reached.
        downscale_factor = 1
        while downscale_factor <= max_downscale_factor:
            b, num_ip_adapters, h, w = mask_tensor.shape
            assert b == 1 and num_ip_adapters == 1

            # The IP-Adapters are applied in the cross-attention layers, where the query sequence length is the h * w of
            # the spatial features.
            query_seq_len = h * w

            masks_by_seq_len[query_seq_len] = mask_tensor.view((b, num_ip_adapters, -1, 1))

            downscale_factor *= 2
            if downscale_factor <= max_downscale_factor:
                # We use max pooling because we downscale to a pretty low resolution, so we don't want small mask
                # regions to be lost entirely.
                #
                # ceil_mode=True is set to mirror the downsampling behavior of SD and SDXL.
                #
                # TODO(ryand): In the future, we may want to experiment with other downsampling methods.
                mask_tensor = torch.nn.functional.max_pool2d(mask_tensor, kernel_size=2, stride=2, ceil_mode=True)

        return masks_by_seq_len


    @callback(CallbackApi.post_run_attention)
    def post_run_attention(self, ctx: AttentionContext, denoise_ctx: DenoiseContext):
        # skip if no weights for this attention
        if str(ctx.module_id) not in self.model.attn_weights._weights:
            return

        # skip if adapter not marked to work on this attention
        if not any(target_block in ctx.module_key for target_block in self.target_blocks):
            return

        # skip if model not active in current step
        total_steps = len(denoise_ctx.inputs.timesteps)
        first_step = math.floor(self.begin_step_percent * total_steps)
        last_step = math.ceil(self.end_step_percent * total_steps)
        if denoise_ctx.step_index < first_step or denoise_ctx.step_index > last_step:
            return

        weight = self.weight
        if isinstance(weight, List):
            weight = weight[denoise_ctx.step_index]

        if weight == 0:
            return

        if denoise_ctx.conditioning_mode == ConditioningMode.Both:
            embeds = torch.stack(
                [self.conditioning.uncond_image_prompt_embeds, self.conditioning.cond_image_prompt_embeds]
            )
        elif denoise_ctx.conditioning_mode == ConditioningMode.Negative:
            embeds = torch.stack([self.conditioning.uncond_image_prompt_embeds])
        else:  # elif denoise_ctx.conditioning_mode == ConditioningMode.Positive:
            embeds = torch.stack([self.conditioning.cond_image_prompt_embeds])

        ip_attn_weights = self.model.attn_weights.get_attention_processor_weights(ctx.module_id)
        ip_key = ip_attn_weights.to_k_ip(embeds)
        ip_value = ip_attn_weights.to_v_ip(embeds)

        ip_hidden_states = ctx.attention_processor.run_attention(
            attn=ctx.attn,
            query=ctx.query,
            key=ip_key,
            value=ip_value,
            attention_mask=None,
        )

        mask = self.mask_tensor[ctx.hidden_states.shape[1]].squeeze(0)

        # Expected ip_hidden_states shape: (batch_size, query_seq_len, num_heads * head_dim)
        ctx.hidden_states += weight * ip_hidden_states * mask
