from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional
from dataclasses import dataclass

import copy
import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput

from invokeai.app.invocations.constants import LATENT_SCALE_FACTOR
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback, override
from invokeai.backend.stable_diffusion.extensions_manager import CallbackApi, OverrideApi
from invokeai.backend.tiles.tiles import calc_tiles_min_overlap
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager

@dataclass
class FakeMultiDiffusionSchedulerOutput(SchedulerOutput):  # BaseOutput
    # prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None

class MultiDiffusionExt(ExtensionBase):
    def __init__(self, tile_height: int, tile_width: int, tile_overlap: int):
        super().__init__()

        self.tile_height = tile_height
        self.tile_width = tile_width
        self.tile_overlap = tile_overlap

        self.tiles = None
        self.region_batch_schedulers = None

    @callback(CallbackApi.pre_denoise_loop)
    def init_tiles(self, ctx: DenoiseContext):
        _, _, latent_height, latent_width = ctx.latents.shape
        latent_tile_height = self.tile_height // LATENT_SCALE_FACTOR
        latent_tile_width = self.tile_width // LATENT_SCALE_FACTOR
        latent_tile_overlap = self.tile_overlap // LATENT_SCALE_FACTOR

        self.tiles = calc_tiles_min_overlap(
            image_height=latent_height,
            image_width=latent_width,
            tile_height=latent_tile_height,
            tile_width=latent_tile_width,
            min_overlap=latent_tile_overlap,
        )
        self.region_batch_schedulers: List[SchedulerMixin] = [copy.deepcopy(ctx.scheduler) for _ in self.tiles]

    @override(OverrideApi.step)
    def tiled_step(self, orig_function: Callable[[DenoiseContext], SchedulerOutput], ctx: DenoiseContext):
        batch_size, _, latent_height, latent_width = ctx.latents.shape

        merged_latents = torch.zeros_like(ctx.latents)
        merged_latents_weights = torch.zeros(
            (1, 1, latent_height, latent_width), device=ctx.latents.device, dtype=ctx.latents.dtype
        )
        merged_pred_original: torch.Tensor | None = None
        for region_idx, region in enumerate(self.tiles):
            print(f"{region_idx=} {region=}")
            region_ctx = DenoiseContext(**vars(ctx))

            # Crop the inputs to the region.
            region_ctx.latents = ctx.latents[
                :,
                :,
                region.coords.top : region.coords.bottom,
                region.coords.left : region.coords.right,
            ]

            # Switch to the scheduler for the region batch.
            region_ctx.scheduler = self.region_batch_schedulers[region_idx]

            # region_ctx.conditioning_data = region_conditioning.text_conditioning_data
            region_ctx.extra["tile_coords"] = region.coords

            # Run the denoising step on the region.
            step_output = orig_function(region_ctx)

            # Build a region_weight matrix that applies gradient blending to the edges of the region.
            _, _, region_height, region_width = step_output.prev_sample.shape
            region_weight = torch.ones(
                (1, 1, region_height, region_width),
                dtype=ctx.latents.dtype,
                device=ctx.latents.device,
            )
            if region.overlap.left > 0:
                left_grad = torch.linspace(
                    0, 1, region.overlap.left, device=ctx.latents.device, dtype=ctx.latents.dtype
                ).view((1, 1, 1, -1))
                region_weight[:, :, :, : region.overlap.left] *= left_grad
            if region.overlap.top > 0:
                top_grad = torch.linspace(
                    0, 1, region.overlap.top, device=ctx.latents.device, dtype=ctx.latents.dtype
                ).view((1, 1, -1, 1))
                region_weight[:, :, : region.overlap.top, :] *= top_grad
            if region.overlap.right > 0:
                right_grad = torch.linspace(
                    1, 0, region.overlap.right, device=ctx.latents.device, dtype=ctx.latents.dtype
                ).view((1, 1, 1, -1))
                region_weight[:, :, :, -region.overlap.right :] *= right_grad
            if region.overlap.bottom > 0:
                bottom_grad = torch.linspace(
                    1, 0, region.overlap.bottom, device=ctx.latents.device, dtype=ctx.latents.dtype
                ).view((1, 1, -1, 1))
                region_weight[:, :, -region.overlap.bottom :, :] *= bottom_grad

            # Update the merged results with the region results.
            merged_latents[
                :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
            ] += step_output.prev_sample * region_weight
            merged_latents_weights[
                :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
            ] += region_weight

            pred_orig_sample = getattr(step_output, "pred_original_sample", None)
            if pred_orig_sample is not None:
                # If one region has pred_original_sample, then we can assume that all regions will have it, because
                # they all use the same scheduler.
                if merged_pred_original is None:
                    merged_pred_original = torch.zeros_like(ctx.latents)
                merged_pred_original[
                    :, :, region.coords.top : region.coords.bottom, region.coords.left : region.coords.right
                ] += pred_orig_sample

        # Normalize the merged results.
        latents = torch.where(merged_latents_weights > 0, merged_latents / merged_latents_weights, merged_latents)
        # For debugging, uncomment this line to visualize the region seams:
        # latents = torch.where(merged_latents_weights > 1, 0.0, latents)
        predicted_original = None
        if merged_pred_original is not None:
            predicted_original = torch.where(
                merged_latents_weights > 0, merged_pred_original / merged_latents_weights, merged_pred_original
            )

        return FakeMultiDiffusionSchedulerOutput(
            prev_sample=latents,
            pred_original_sample=predicted_original,
        )

