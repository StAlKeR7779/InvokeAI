from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState

if TYPE_CHECKING:
    from invokeai.backend.flux.denoise_context import DenoiseContext


class PreviewExt:
    def __init__(self, callback: Callable[[PipelineIntermediateState], None]):
        super().__init__()
        self.callback = callback

    def initial_preview(self, ctx: DenoiseContext):
        self.callback(
            PipelineIntermediateState(
                step=0,
                order=ctx.scheduler.order,
                total_steps=ctx.total_steps,
                timestep=int(ctx.scheduler.timesteps[0]),  # TODO: is there any code which uses it?
                latents=ctx.img,
            )
        )

    def step_preview(self, ctx: DenoiseContext):
        if ctx.is_scheduler_internal_step:
            return

        if hasattr(ctx.step_output, "pred_original_sample"):
            predicted_original = ctx.step_output.pred_original_sample
        else:
            predicted_original = ctx.step_output

        # Only call step_callback if we haven't exceeded total_steps
        # (LCM scheduler may have more internal steps than user-facing steps)
        if ctx.user_step < ctx.total_steps:
            self.callback(
                PipelineIntermediateState(
                    step=ctx.user_step,
                    order=ctx.scheduler.order, # PATCH: use order defined in scheduler class
                    total_steps=ctx.total_steps,
                    timestep=int(ctx.t_curr * 1000),  # TODO: not used anywhere in code
                    latents=predicted_original,
                ),
            )
