from __future__ import annotations

import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput
from tqdm.auto import tqdm

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext, UNetKwargs
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extension_override_type import ExtensionOverrideType
from invokeai.backend.stable_diffusion.extensions_manager import ExtensionsManager


class StableDiffusionBackend:
    def __init__(self):
        config = get_config()
        self._sequential_guidance = config.sequential_guidance

    def latents_from_embeddings(self, ctx: DenoiseContext, ext_manager: ExtensionsManager):
        if ctx.inputs.init_timestep.shape[0] == 0:
            return ctx.inputs.orig_latents

        ctx.latents = ctx.inputs.orig_latents.clone()

        if ctx.inputs.noise is not None:
            batch_size = ctx.latents.shape[0]
            # latents = noise * ctx.scheduler.init_noise_sigma # it's like in t2l according to diffusers
            ctx.latents = ctx.scheduler.add_noise(
                ctx.latents, ctx.inputs.noise, ctx.inputs.init_timestep.expand(batch_size)
            )

        # if no work to do, return latents
        if ctx.inputs.timesteps.shape[0] == 0:
            return ctx.latents

        # ext: inpaint[pre_denoise_loop, priority=normal] (maybe init, but not sure if it needed)
        # ext: preview[pre_denoise_loop, priority=low]
        ext_manager.run_callback(ExtensionCallbackType.PRE_DENOISE_LOOP, ctx)

        for ctx.step_index, ctx.timestep in enumerate(tqdm(ctx.inputs.timesteps)):  # noqa: B020
            # ext: inpaint (apply mask to latents on non-inpaint models)
            ext_manager.run_callback(ExtensionCallbackType.PRE_STEP, ctx)

            # ext: tiles? [override: step]
            ctx.step_output = ext_manager.run_override(ExtensionOverrideType.STEP, self.step, ctx, ext_manager)

            # ext: inpaint[post_step, priority=high] (apply mask to preview on non-inpaint models)
            # ext: preview[post_step, priority=low]
            ext_manager.run_callback(ExtensionCallbackType.POST_STEP, ctx)

            ctx.latents = ctx.step_output.prev_sample

        # ext: inpaint[post_denoise_loop] (restore unmasked part)
        ext_manager.run_callback(ExtensionCallbackType.POST_DENOISE_LOOP, ctx)
        return ctx.latents

    @torch.inference_mode()
    def step(self, ctx: DenoiseContext, ext_manager: ExtensionsManager) -> SchedulerOutput:
        ctx.latent_model_input = ctx.scheduler.scale_model_input(ctx.latents, ctx.timestep)

        # TODO: conditionings as list(conditioning_data.to_unet_kwargs - ready)
        # Note: The current handling of conditioning doesn't feel very future-proof.
        # This might change in the future as new requirements come up, but for now,
        # this is the rough plan.
        if self._sequential_guidance:
            ctx.negative_noise_pred = self.run_unet(ctx, ext_manager, ConditioningMode.Negative)
            ctx.positive_noise_pred = self.run_unet(ctx, ext_manager, ConditioningMode.Positive)
        else:
            both_noise_pred = self.run_unet(ctx, ext_manager, ConditioningMode.Both)
            ctx.negative_noise_pred, ctx.positive_noise_pred = both_noise_pred.chunk(2)

        # ext: override combine_noise_preds
        ctx.noise_pred = ext_manager.run_override(
            ExtensionOverrideType.COMBINE_NOISE_PREDS, self.combine_noise_preds, ctx, ext_manager
        )

        # ext: cfg_rescale [modify_noise_prediction]
        # TODO: rename
        ext_manager.run_callback(ExtensionCallbackType.POST_COMBINE_NOISE_PREDS, ctx)

        # compute the previous noisy sample x_t -> x_t-1
        step_output = ctx.scheduler.step(ctx.noise_pred, ctx.timestep, ctx.latents, **ctx.inputs.scheduler_step_kwargs)

        # clean up locals
        ctx.latent_model_input = None
        ctx.negative_noise_pred = None
        ctx.positive_noise_pred = None
        ctx.noise_pred = None

        return step_output

    def run_unet(self, ctx: DenoiseContext, ext_manager: ExtensionsManager, conditioning_mode: ConditioningMode):
        sample = ctx.latent_model_input
        if conditioning_mode == ConditioningMode.Both:
            sample = torch.cat([sample] * 2)

        ctx.unet_kwargs = UNetKwargs(
            sample=sample,
            timestep=ctx.timestep,
            encoder_hidden_states=None,  # set later by conditoning
            cross_attention_kwargs=dict(  # noqa: C408
                percent_through=ctx.step_index / len(ctx.inputs.timesteps),
            ),
        )

        ctx.conditioning_mode = conditioning_mode
        ctx.inputs.conditioning_data.to_unet_kwargs(ctx.unet_kwargs, ctx.conditioning_mode)

        # ext: controlnet/ip/t2i [pre_unet_forward]
        ext_manager.run_callback(ExtensionCallbackType.PRE_UNET_FORWARD, ctx)

        # ext: inpaint model/ic-light [override: unet_forward]
        noise_pred = ext_manager.run_override(ExtensionOverrideType.UNET_FORWARD, self.unet_forward, ctx, ext_manager)

        ext_manager.run_callback(ExtensionCallbackType.POST_UNET_FORWARD, ctx)

        # clean up locals
        ctx.unet_kwargs = None
        ctx.conditioning_mode = None

        return noise_pred

    # pass extensions manager as arg to allow override access it
    def combine_noise_preds(self, ctx: DenoiseContext, ext_manager: ExtensionsManager) -> torch.Tensor:
        guidance_scale = ctx.inputs.conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[ctx.step_index]

        # Note: Although this `torch.lerp(...)` line is logically equivalent to the current CFG line, it seems to result
        # in slightly different outputs. It is suspected that this is caused by small precision differences.
        # return torch.lerp(ctx.negative_noise_pred, ctx.positive_noise_pred, guidance_scale)
        return ctx.negative_noise_pred + guidance_scale * (ctx.positive_noise_pred - ctx.negative_noise_pred)

    # pass extensions manager as arg to allow override access it
    def unet_forward(self, ctx: DenoiseContext, ext_manager: ExtensionsManager) -> torch.Tensor:
        return ctx.unet(**vars(ctx.unet_kwargs)).sample
