from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback, override
from invokeai.backend.stable_diffusion.extensions_manager import CallbackApi, OverrideApi

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.stable_diffusion.diffusion.custom_attention_new import AttentionContext


# Gaussian blur
def gaussian_blur_2d(img, kernel_size, sigma):
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img

class SEGExt(ExtensionBase):
    def __init__(self, scale: float, blur_sigma: float):
        super().__init__()
        self.do_cfg = True
        self.blur_sigma = blur_sigma
        self.scale = scale

        # inf_blur_threshold=9999.0
        self.inf_blur = self.blur_sigma > 9999.0
        self._enabled = False
        self._index = 0

    #@callback(CallbackApi.pre_unet_forward)
    #def get_hw(self, ctx: DenoiseContext):
    #    _, _, self.height, self.width = ctx.unet_kwargs.sample.shape
    #    ctx.unet_kwargs.sample = torch.cat([ctx.unet_kwargs.sample, ctx.unet_kwargs.sample[-1:]])
    #    print(f"pre: {ctx.unet_kwargs.encoder_hidden_states.shape}")
    #    ctx.unet_kwargs.encoder_hidden_states = torch.cat([ctx.unet_kwargs.encoder_hidden_states, ctx.unet_kwargs.encoder_hidden_states[-1:]])
    #    print(f"post: {ctx.unet_kwargs.encoder_hidden_states.shape}")
    #    if ctx.unet_kwargs.encoder_attention_mask is not None:
    #        ctx.unet_kwargs.encoder_attention_mask = torch.cat([ctx.unet_kwargs.encoder_attention_mask, ctx.unet_kwargs.encoder_attention_mask[-1:]])

    @override(OverrideApi.combine_noise_preds)
    def combine_noise_preds(self, orig_function: Callable[[DenoiseContext], torch.Tensor], ctx: DenoiseContext):
        try:
            self._enabled = True
            self._index = 0
            _, _, self.height, self.width = ctx.latents.shape
            seg_noise_pred = ctx.backend.run_unet(ctx, ConditioningMode.Positive)
        finally:
            self._enabled = False

        return orig_function(ctx) + self.scale * (ctx.positive_noise_pred - seg_noise_pred)

    @callback(CallbackApi.pre_run_attention)
    def do_seg_magic(self, ctx: AttentionContext, denoise_ctx: DenoiseContext):
        if not self._enabled:
            return

        # TODO: normal filtering
        if "mid" not in ctx.processor.attention_key:
            return

        # orig, looks bad
        if False:
            height = width = math.isqrt(ctx.query.shape[1])
        else:
            # ctx.query.shape[1] = (height * width) / scale**2
            hw_scale = math.isqrt((self.width * self.height) // ctx.query.shape[1])
            width = self.width // hw_scale
            height = self.height // hw_scale

        batch_size, _, seq_len = ctx.query.shape

        # TODO: batch
        query_ptb = ctx.query[self._index].unsqueeze(0)
        query_ptb = query_ptb.permute(0, 2, 1).view(1, seq_len, height, width)

        if not self.inf_blur:
            kernel_size = math.ceil(6 * self.blur_sigma) + 1 - math.ceil(6 * self.blur_sigma) % 2
            query_ptb = gaussian_blur_2d(query_ptb, kernel_size, self.blur_sigma)
        else:
            query_ptb[:] = query_ptb.mean(dim=(-2, -1), keepdim=True)

        query_ptb = query_ptb.view(1, seq_len, height * width).permute(0, 2, 1)
        ctx.query[self._index] = query_ptb
