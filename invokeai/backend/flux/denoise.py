import inspect
import math
import contextlib
from typing import Callable
from dataclasses import dataclass, field

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm

from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput, sum_controlnet_flux_outputs
from invokeai.backend.flux.extensions.dype_extension import DyPEExtension
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.extensions.xlabs_controlnet_extension import XLabsControlNetExtension
from invokeai.backend.flux.extensions.xlabs_ip_adapter_extension import XLabsIPAdapterExtension
from invokeai.backend.flux.model import Flux
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.util.devices import TorchDevice

from invokeai.backend.flux.denoise_context import DenoiseContext
# TODO: move outside sd
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ConditioningMode


def denoise(
    model: Flux,
    # model input
    img: torch.Tensor,
    img_ids: torch.Tensor,
    pos_regional_prompting_extension: RegionalPromptingExtension,
    neg_regional_prompting_extension: RegionalPromptingExtension | None,
    # sampling parameters
    timesteps: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    guidance: float,
    cfg_scale: list[float],
    inpaint_extension: RectifiedFlowInpaintExtension | None,
    controlnet_extensions: list[XLabsControlNetExtension | InstantXControlNetExtension],
    pos_ip_adapter_extensions: list[XLabsIPAdapterExtension],
    neg_ip_adapter_extensions: list[XLabsIPAdapterExtension],
    # extra img tokens (channel-wise)
    img_cond: torch.Tensor | None,
    # extra img tokens (sequence-wise) - for Kontext conditioning
    img_cond_seq: torch.Tensor | None = None,
    img_cond_seq_ids: torch.Tensor | None = None,
    # DyPE extension for high-resolution generation
    dype_extension: DyPEExtension | None = None,
    # Optional scheduler for alternative sampling methods
    scheduler: SchedulerMixin | None = None,
):
    ctx = DenoiseContext(
        model=model,
        # model input
        img=img,
        img_ids=img_ids,
        pos_regional_prompting_extension=pos_regional_prompting_extension,
        neg_regional_prompting_extension=neg_regional_prompting_extension,
        # sampling parameters
        timesteps=timesteps,
        scheduler=scheduler,
        step_callback=step_callback,
        guidance=guidance,
        cfg_scale=cfg_scale,
        inpaint_extension=inpaint_extension,
        controlnet_extensions=controlnet_extensions,
        pos_ip_adapter_extensions=pos_ip_adapter_extensions,
        neg_ip_adapter_extensions=neg_ip_adapter_extensions,
        # extra img tokens (channel-wise)
        img_cond=img_cond,
        # extra img tokens (sequence-wise) - for Kontext conditioning
        img_cond_seq=img_cond_seq,
        img_cond_seq_ids=img_cond_seq_ids,
        # DyPE extension for high-resolution generation
        dype_extension=dype_extension,
    )
    with contextlib.ExitStack() as exit_stack:
        if dype_extension is not None:
            exit_stack.enter_context(dype_extension.patch_model(ctx, ctx.model))
        return denoise_(
            ctx=ctx,
        )


def denoise_(
    ctx: DenoiseContext,
):
    assert ctx.scheduler is not None

    # Initialize scheduler with timesteps
    # The timesteps list contains values in [0, 1] range (sigmas)
    # LCM should use num_inference_steps (it has its own sigma schedule),
    # while other schedulers can use custom sigmas if supported
    is_lcm = ctx.scheduler.__class__.__name__ == "FlowMatchLCMScheduler"
    set_timesteps_sig = inspect.signature(ctx.scheduler.set_timesteps)
    if not is_lcm and "sigmas" in set_timesteps_sig.parameters:
        # Scheduler supports custom sigmas - use InvokeAI's time-shifted schedule
        ctx.scheduler.set_timesteps(sigmas=ctx.timesteps, device=ctx.img.device)
    else:
        # LCM or scheduler doesn't support custom sigmas - use num_inference_steps
        # The schedule will be computed by the scheduler itself.
        #
        # Important for img2img callers: if the initial latent/noise blend was
        # computed from a separate pre-scheduler schedule, that preblend may not
        # match this scheduler's true first step exactly.
        num_inference_steps = len(ctx.timesteps) - 1
        ctx.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=ctx.img.device)

    # For schedulers like Heun, the number of actual steps may differ
    # (Heun doubles timesteps internally)
    num_scheduler_steps = len(ctx.scheduler.timesteps)
    # For user-facing step count, use the original number of denoising steps
    ctx.total_steps = len(ctx.timesteps) - 1

    # Track the actual step for user-facing progress (accounts for Heun's double steps)
    ctx.user_step = 0

    # Use tqdm with total_steps (user-facing steps) not num_scheduler_steps (internal steps)
    # This ensures progress bar shows 1/8, 2/8, etc. even when scheduler uses more internal steps
    pbar = tqdm(total=ctx.total_steps, desc=f"Denoising{TorchDevice.get_session_device_label()}")
    for ctx.step_index in range(num_scheduler_steps):
        timestep = ctx.scheduler.timesteps[ctx.step_index]
        # Convert scheduler timestep (0-1000) to normalized (0-1) for the model
        ctx.t_curr = timestep.item() / ctx.scheduler.config.num_train_timesteps

        # PRE_SAMPLER_STEP - DyPEExtension
        # DyPE: Update step state for timestep-dependent scaling
        if ctx.dype_extension is not None:
            ctx.dype_extension.update_step_state(ctx)

        step_cfg_scale = ctx.cfg_scale[min(ctx.user_step, len(ctx.cfg_scale) - 1)]
        if math.isclose(step_cfg_scale, 1.0):
            pred = guidance_none(ctx)
        else:
            pred = guidance_cfg(ctx, step_cfg_scale)

        # Use scheduler.step() for the update
        step_output = ctx.scheduler.step(model_output=pred, timestep=timestep, sample=ctx.img)
        ctx.img = step_output.prev_sample

        # POST_SAMPLER_STEP -  RectifiedFlowInpaintExtension, PreviewExt(order=last)
        if ctx.inpaint_extension is not None:
            # Get sigma_prev for inpainting (next sigma value)
            if ctx.step_index + 1 < len(ctx.scheduler.sigmas):
                sigma_prev = ctx.scheduler.sigmas[ctx.step_index + 1].item()
            else:
                sigma_prev = 0.0
            ctx.img = ctx.inpaint_extension.merge_intermediate_latents_with_init_latents(ctx.img, sigma_prev)

        # For Heun, only increment user step after second-order step completes
        is_heun = hasattr(ctx.scheduler, "state_in_first_order")
        in_first_order = ctx.scheduler.state_in_first_order if is_heun else True
        if (is_heun and not in_first_order) or (not is_heun):
            ctx.user_step += 1
            # Only call step_callback if we haven't exceeded total_steps
            # (LCM scheduler may have more internal steps than user-facing steps)
            if ctx.user_step <= ctx.total_steps:
                pbar.update(1)
                preview_img = ctx.img - ctx.t_curr * pred
                if ctx.inpaint_extension is not None:
                    preview_img = ctx.inpaint_extension.merge_intermediate_latents_with_init_latents(
                        preview_img, 0.0
                    )
                ctx.step_callback(
                    PipelineIntermediateState(
                        step=ctx.user_step,
                        order=2 if is_heun else 1,
                        total_steps=ctx.total_steps,
                        timestep=int(ctx.t_curr * 1000),  # TODO: not used anywhere in code
                        latents=preview_img,
                    ),
                )

    pbar.close()
    return ctx.img


def guidance_none(ctx: DenoiseContext) -> torch.Tensor:
    return run_model(ctx, ConditioningMode.Positive)


def guidance_cfg(ctx: DenoiseContext, step_cfg_scale: float) -> torch.Tensor:
    if ctx.neg_regional_prompting_extension is None:
        raise ValueError("Negative text conditioning is required when cfg_scale is not 1.0.")

    pos_pred = run_model(ctx, ConditioningMode.Positive)
    neg_pred = run_model(ctx, ConditioningMode.Negative)

    pred = neg_pred + step_cfg_scale * (pos_pred - neg_pred)
    return pred


def run_controlnets(ctx: DenoiseContext):
    assert ctx.conditioning_mode != ConditioningMode.Both
    if ctx.conditioning_mode == ConditioningMode.Positive:
        regional_prompting_extension = ctx.pos_regional_prompting_extension
    else:
        regional_prompting_extension = ctx.neg_regional_prompting_extension

    # TODO:
    if ctx.conditioning_mode != ConditioningMode.Positive:
        return None

    # Run ControlNet models.
    t_vec = torch.full((ctx.img.shape[0],), ctx.t_curr, dtype=ctx.img.dtype, device=ctx.img.device)
    guidance_vec = torch.full((ctx.img.shape[0],), ctx.guidance, device=ctx.img.device, dtype=ctx.img.dtype)
    controlnet_residuals: list[ControlNetFluxOutput] = []
    for controlnet_extension in ctx.controlnet_extensions:
        controlnet_residuals.append(
            controlnet_extension.run_controlnet(
                timestep_index=ctx.user_step,  # TODO:
                total_num_timesteps=ctx.total_steps,
                img=ctx.img,
                img_ids=ctx.img_ids,
                txt=regional_prompting_extension.regional_text_conditioning.t5_embeddings,
                txt_ids=regional_prompting_extension.regional_text_conditioning.t5_txt_ids,
                y=regional_prompting_extension.regional_text_conditioning.clip_embeddings,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
        )

    # Merge the ControlNet residuals from multiple ControlNets.
    # TODO(ryand): We may want to calculate the sum just-in-time to keep peak memory low. Keep in mind, that the
    # controlnet_residuals datastructure is efficient in that it likely contains multiple references to the same
    # tensors. Calculating the sum materializes each tensor into its own instance.
    merged_controlnet_residuals = sum_controlnet_flux_outputs(controlnet_residuals)
    return merged_controlnet_residuals


def run_model(ctx: DenoiseContext, conditioning_mode: ConditioningMode):
    assert conditioning_mode != ConditioningMode.Both  # TODO: batch
    if conditioning_mode == ConditioningMode.Positive:
        regional_prompting_extension = ctx.pos_regional_prompting_extension
        ip_adapter_extensions = ctx.pos_ip_adapter_extensions
    else:
        regional_prompting_extension = ctx.neg_regional_prompting_extension
        ip_adapter_extensions = ctx.neg_ip_adapter_extensions

    ctx.conditioning_mode = conditioning_mode

    # Store original sequence length for slicing predictions
    original_seq_len = ctx.img.shape[1]

    # Prepare input for model - concatenate fresh each step
    img_input = ctx.img
    img_input_ids = ctx.img_ids

    # Add channel-wise conditioning (for ControlNet, FLUX Fill, etc.)
    if ctx.img_cond is not None:
        img_input = torch.cat((img_input, ctx.img_cond), dim=-1)

    # Add sequence-wise conditioning (for Kontext)
    if ctx.img_cond_seq is not None:
        assert ctx.img_cond_seq_ids is not None, (
            "You need to provide either both or neither of the sequence conditioning"
        )
        img_input = torch.cat((img_input, ctx.img_cond_seq), dim=1)
        img_input_ids = torch.cat((img_input_ids, ctx.img_cond_seq_ids), dim=1)

    # PRE_MODEL_RUN - XLabsControlNetExtension | InstantXControlNetExtension
    # Run ControlNet models.
    merged_controlnet_residuals = run_controlnets(ctx)

    controlnet_double_block_residuals = None
    controlnet_single_block_residuals = None
    if (merged_controlnet_residuals != None):
        controlnet_double_block_residuals = merged_controlnet_residuals.double_block_residuals
        controlnet_single_block_residuals = merged_controlnet_residuals.single_block_residuals

    t_vec = torch.full((ctx.img.shape[0],), ctx.t_curr, dtype=ctx.img.dtype, device=ctx.img.device)
    guidance_vec = torch.full((ctx.img.shape[0],), ctx.guidance, device=ctx.img.device, dtype=ctx.img.dtype)
    pred = ctx.model(
        img=img_input,
        img_ids=img_input_ids,
        txt=regional_prompting_extension.regional_text_conditioning.t5_embeddings,
        txt_ids=regional_prompting_extension.regional_text_conditioning.t5_txt_ids,
        y=regional_prompting_extension.regional_text_conditioning.clip_embeddings,
        timesteps=t_vec,
        guidance=guidance_vec,
        timestep_index=ctx.user_step,  # TODO:
        total_num_timesteps=ctx.total_steps,
        controlnet_double_block_residuals=controlnet_double_block_residuals,
        controlnet_single_block_residuals=controlnet_single_block_residuals,
        ip_adapter_extensions=ip_adapter_extensions,
        regional_prompting_extension=regional_prompting_extension,
    )

    # Slice prediction to only include the main image tokens
    if ctx.img_cond_seq is not None:
        pred = pred[:, :original_seq_len]

    ctx.conditioning_mode = None

    return pred
