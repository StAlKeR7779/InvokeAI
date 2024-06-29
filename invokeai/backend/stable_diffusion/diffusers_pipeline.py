from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import einops
import PIL.Image
import psutil
import torch
import torchvision.transforms as T
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin
from diffusers.utils.import_utils import is_xformers_available
from pydantic import Field
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import IPAdapterData, TextConditioningData
from invokeai.backend.stable_diffusion.diffusion.shared_invokeai_diffusion import InvokeAIDiffuserComponent
from invokeai.backend.stable_diffusion.diffusion.unet_attention_patcher import UNetAttentionPatcher, UNetIPAdapterData
from invokeai.backend.util.attention import auto_detect_slice_size
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.hotfixes import ControlNetModel


@dataclass
class PipelineIntermediateState:
    step: int
    order: int
    total_steps: int
    timestep: int
    latents: torch.Tensor
    predicted_original: Optional[torch.Tensor] = None


@dataclass
class AddsMaskGuidance:
    mask: torch.Tensor
    mask_latents: torch.Tensor
    scheduler: SchedulerMixin
    noise: torch.Tensor
    is_gradient_mask: bool

    def __call__(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.apply_mask(latents, t)

    def apply_mask(self, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = latents.size(0)
        mask = einops.repeat(self.mask, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if t.dim() == 0:
            # some schedulers expect t to be one-dimensional.
            # TODO: file diffusers bug about inconsistency?
            t = einops.repeat(t, "-> batch", batch=batch_size)
        # Noise shouldn't be re-randomized between steps here. The multistep schedulers
        # get very confused about what is happening from step to step when we do that.
        mask_latents = self.scheduler.add_noise(self.mask_latents, self.noise, t)
        # TODO: Do we need to also apply scheduler.scale_model_input? Or is add_noise appropriately scaled already?
        # mask_latents = self.scheduler.scale_model_input(mask_latents, t)
        mask_latents = einops.repeat(mask_latents, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if self.is_gradient_mask:
            threshhold = (t.item()) / self.scheduler.config.num_train_timesteps
            mask_bool = mask > threshhold  # I don't know when mask got inverted, but it did
            masked_input = torch.where(mask_bool, latents, mask_latents)
        else:
            masked_input = torch.lerp(mask_latents.to(dtype=latents.dtype), latents, mask.to(dtype=latents.dtype))
        return masked_input


def trim_to_multiple_of(*args, multiple_of=8):
    return tuple((x - x % multiple_of) for x in args)


def image_resized_to_grid_as_tensor(image: PIL.Image.Image, normalize: bool = True, multiple_of=8) -> torch.FloatTensor:
    """

    :param image: input image
    :param normalize: scale the range to [-1, 1] instead of [0, 1]
    :param multiple_of: resize the input so both dimensions are a multiple of this
    """
    w, h = trim_to_multiple_of(*image.size, multiple_of=multiple_of)
    transformation = T.Compose(
        [
            T.Resize((h, w), T.InterpolationMode.LANCZOS, antialias=True),
            T.ToTensor(),
        ]
    )
    tensor = transformation(image)
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def is_inpainting_model(unet: UNet2DConditionModel):
    return unet.conv_in.in_channels == 9


@dataclass
class ControlNetData:
    model: ControlNetModel = Field(default=None)
    image_tensor: torch.Tensor = Field(default=None)
    weight: Union[float, List[float]] = Field(default=1.0)
    begin_step_percent: float = Field(default=0.0)
    end_step_percent: float = Field(default=1.0)
    control_mode: str = Field(default="balanced")
    resize_mode: str = Field(default="just_resize")


@dataclass
class T2IAdapterData:
    """A structure containing the information required to apply conditioning from a single T2I-Adapter model."""

    adapter_state: dict[torch.Tensor] = Field()
    weight: Union[float, list[float]] = Field(default=1.0)
    begin_step_percent: float = Field(default=0.0)
    end_step_percent: float = Field(default=1.0)


class StableDiffusionGeneratorPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Implementation note: This class started as a refactored copy of diffusers.StableDiffusionPipeline.
    Hopefully future versions of diffusers provide access to more of these functions so that we don't
    need to duplicate them here: https://github.com/huggingface/diffusers/issues/551#issuecomment-1281508384

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: Optional[StableDiffusionSafetyChecker],
        feature_extractor: Optional[CLIPFeatureExtractor],
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker,
        )

        self.invokeai_diffuser = InvokeAIDiffuserComponent(self.unet, self._unet_forward)

    def _adjust_memory_efficient_attention(self, latents: torch.Tensor):
        """
        if xformers is available, use it, otherwise use sliced attention.
        """
        config = get_config()
        if config.attention_type == "xformers":
            self.enable_xformers_memory_efficient_attention()
            return
        elif config.attention_type == "sliced":
            slice_size = config.attention_slice_size
            if slice_size == "auto":
                slice_size = auto_detect_slice_size(latents)
            elif slice_size == "balanced":
                slice_size = "auto"
            self.enable_attention_slicing(slice_size=slice_size)
            return
        elif config.attention_type == "normal":
            self.disable_attention_slicing()
            return
        elif config.attention_type == "torch-sdp":
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                # diffusers enables sdp automatically
                return
            else:
                raise Exception("torch-sdp attention slicing not available")

        # the remainder if this code is called when attention_type=='auto'
        if self.unet.device.type == "cuda":
            if is_xformers_available():
                self.enable_xformers_memory_efficient_attention()
                return
            elif hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                # diffusers enables sdp automatically
                return

        if self.unet.device.type == "cpu" or self.unet.device.type == "mps":
            mem_free = psutil.virtual_memory().free
        elif self.unet.device.type == "cuda":
            mem_free, _ = torch.cuda.mem_get_info(TorchDevice.normalize(self.unet.device))
        else:
            raise ValueError(f"unrecognized device {self.unet.device}")
        # input tensor of [1, 4, h/8, w/8]
        # output tensor of [16, (h/8 * w/8), (h/8 * w/8)]
        bytes_per_element_needed_for_baddbmm_duplication = latents.element_size() + 4
        max_size_required_for_baddbmm = (
            16
            * latents.size(dim=2)
            * latents.size(dim=3)
            * latents.size(dim=2)
            * latents.size(dim=3)
            * bytes_per_element_needed_for_baddbmm_duplication
        )
        if max_size_required_for_baddbmm > (mem_free * 3.0 / 4.0):  # 3.3 / 4.0 is from old Invoke code
            self.enable_attention_slicing(slice_size="max")
        elif torch.backends.mps.is_available():
            # diffusers recommends always enabling for mps
            self.enable_attention_slicing(slice_size="max")
        else:
            self.disable_attention_slicing()

    def to(self, torch_device: Optional[Union[str, torch.device]] = None, silence_dtype_warnings=False):
        raise Exception("Should not be called")

    def add_inpainting_channels_to_latents(
        self, latents: torch.Tensor, masked_ref_image_latents: torch.Tensor, inpainting_mask: torch.Tensor
    ):
        """Given a `latents` tensor, adds the mask and image latents channels required for inpainting.

        Standard (non-inpainting) SD UNet models expect an input with shape (N, 4, H, W). Inpainting models expect an
        input of shape (N, 9, H, W). The 9 channels are defined as follows:
        - Channel 0-3: The latents being denoised.
        - Channel 4: The mask indicating which parts of the image are being inpainted.
        - Channel 5-8: The latent representation of the masked reference image being inpainted.

        This function assumes that the same mask and base image should apply to all items in the batch.
        """
        # Validate assumptions about input tensor shapes.
        batch_size, latent_channels, latent_height, latent_width = latents.shape
        assert latent_channels == 4
        assert list(masked_ref_image_latents.shape) == [1, 4, latent_height, latent_width]
        assert list(inpainting_mask.shape) == [1, 1, latent_height, latent_width]

        # Repeat original_image_latents and inpainting_mask to match the latents batch size.
        original_image_latents = masked_ref_image_latents.expand(batch_size, -1, -1, -1)
        inpainting_mask = inpainting_mask.expand(batch_size, -1, -1, -1)

        # Concatenate along the channel dimension.
        return torch.cat([latents, inpainting_mask, original_image_latents], dim=1)

    def latents_from_embeddings(
        self,
        latents: torch.Tensor,
        scheduler_step_kwargs: dict[str, Any],
        conditioning_data: TextConditioningData,
        noise: Optional[torch.Tensor],
        seed: int,
        timesteps: torch.Tensor,
        init_timestep: torch.Tensor,
        callback: Callable[[PipelineIntermediateState], None],
        control_data: list[ControlNetData] | None = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
        mask: Optional[torch.Tensor] = None,
        masked_latents: Optional[torch.Tensor] = None,
        is_gradient_mask: bool = False,
    ) -> torch.Tensor:
        """Denoise the latents.

        Args:
            latents: The latent-space image to denoise.
                - If we are inpainting, this is the initial latent image before noise has been added.
                - If we are generating a new image, this should be initialized to zeros.
                - In some cases, this may be a partially-noised latent image (e.g. when running the SDXL refiner).
            scheduler_step_kwargs: kwargs forwarded to the scheduler.step() method.
            conditioning_data: Text conditionging data.
            noise: Noise used for two purposes:
                1. Used by the scheduler to noise the initial `latents` before denoising.
                2. Used to noise the `masked_latents` when inpainting.
                `noise` should be None if the `latents` tensor has already been noised.
            seed: The seed used to generate the noise for the denoising process.
                HACK(ryand): seed is only used in a particular case when `noise` is None, but we need to re-generate the
                same noise used earlier in the pipeline. This should really be handled in a clearer way.
            timesteps: The timestep schedule for the denoising process.
            init_timestep: The first timestep in the schedule. This is used to determine the initial noise level, so
                should be populated if you want noise applied *even* if timesteps is empty.
            callback: A callback function that is called to report progress during the denoising process.
            control_data: ControlNet data.
            ip_adapter_data: IP-Adapter data.
            t2i_adapter_data: T2I-Adapter data.
            mask: A mask indicating which parts of the image are being inpainted. The presence of mask is used to
                determine whether we are inpainting or not. `mask` should have the same spatial dimensions as the
                `latents` tensor.
                TODO(ryand): Check and document the expected dtype, range, and values used to represent
                foreground/background.
            masked_latents: A latent-space representation of a masked inpainting reference image. This tensor is only
                used if an *inpainting* model is being used i.e. this tensor is not used when inpainting with a standard
                SD UNet model.
            is_gradient_mask: A flag indicating whether `mask` is a gradient mask or not.
        """

        if True:
            back = StableDiffusionBackend(self.unet, self.scheduler)
            return back.latents_from_embeddings(
                latents=latents,
                scheduler_step_kwargs=scheduler_step_kwargs,
                conditioning_data=conditioning_data,
                noise=noise,
                seed=seed,
                timesteps=timesteps,
                init_timestep=init_timestep,
                callback=callback,
                control_data=control_data,
                ip_adapter_data=ip_adapter_data,
                t2i_adapter_data=t2i_adapter_data,
                mask=mask,
                masked_latents=masked_latents,
                is_gradient_mask=is_gradient_mask,
            )

        if init_timestep.shape[0] == 0:
            return latents

        orig_latents = latents.clone()

        batch_size = latents.shape[0]
        batched_init_timestep = init_timestep.expand(batch_size)

        # noise can be None if the latents have already been noised (e.g. when running the SDXL refiner).
        if noise is not None:
            # TODO(ryand): I'm pretty sure we should be applying init_noise_sigma in cases where we are starting with
            # full noise. Investigate the history of why this got commented out.
            # latents = noise * self.scheduler.init_noise_sigma # it's like in t2l according to diffusers
            latents = self.scheduler.add_noise(latents, noise, batched_init_timestep)

        self._adjust_memory_efficient_attention(latents)

        # Handle mask guidance (a.k.a. inpainting).
        mask_guidance: AddsMaskGuidance | None = None
        if mask is not None and not is_inpainting_model(self.unet):
            # We are doing inpainting, since a mask is provided, but we are not using an inpainting model, so we will
            # apply mask guidance to the latents.

            # 'noise' might be None if the latents have already been noised (e.g. when running the SDXL refiner).
            # We still need noise for inpainting, so we generate it from the seed here.
            if noise is None:
                noise = torch.randn(
                    orig_latents.shape,
                    dtype=torch.float32,
                    device="cpu",
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                ).to(device=orig_latents.device, dtype=orig_latents.dtype)

            mask_guidance = AddsMaskGuidance(
                mask=mask,
                mask_latents=orig_latents,
                scheduler=self.scheduler,
                noise=noise,
                is_gradient_mask=is_gradient_mask,
            )

        use_ip_adapter = ip_adapter_data is not None
        use_regional_prompting = (
            conditioning_data.cond_regions is not None or conditioning_data.uncond_regions is not None
        )
        unet_attention_patcher = None
        attn_ctx = nullcontext()

        if use_ip_adapter or use_regional_prompting:
            ip_adapters: Optional[List[UNetIPAdapterData]] = (
                [{"ip_adapter": ipa.ip_adapter_model, "target_blocks": ipa.target_blocks} for ipa in ip_adapter_data]
                if use_ip_adapter
                else None
            )
            unet_attention_patcher = UNetAttentionPatcher(ip_adapters)
            attn_ctx = unet_attention_patcher.apply_ip_adapter_attention(self.invokeai_diffuser.model)

        with attn_ctx:
            callback(
                PipelineIntermediateState(
                    step=-1,
                    order=self.scheduler.order,
                    total_steps=len(timesteps),
                    timestep=self.scheduler.config.num_train_timesteps,
                    latents=latents,
                )
            )

            for i, t in enumerate(self.progress_bar(timesteps)):
                batched_t = t.expand(batch_size)
                step_output = self.step(
                    t=batched_t,
                    latents=latents,
                    conditioning_data=conditioning_data,
                    step_index=i,
                    total_step_count=len(timesteps),
                    scheduler_step_kwargs=scheduler_step_kwargs,
                    mask_guidance=mask_guidance,
                    mask=mask,
                    masked_latents=masked_latents,
                    control_data=control_data,
                    ip_adapter_data=ip_adapter_data,
                    t2i_adapter_data=t2i_adapter_data,
                )
                latents = step_output.prev_sample
                predicted_original = getattr(step_output, "pred_original_sample", None)

                callback(
                    PipelineIntermediateState(
                        step=i,
                        order=self.scheduler.order,
                        total_steps=len(timesteps),
                        timestep=int(t),
                        latents=latents,
                        predicted_original=predicted_original,
                    )
                )

        # restore unmasked part after the last step is completed
        # in-process masking happens before each step
        if mask is not None:
            if is_gradient_mask:
                latents = torch.where(mask > 0, latents, orig_latents)
            else:
                latents = torch.lerp(
                    orig_latents, latents.to(dtype=orig_latents.dtype), mask.to(dtype=orig_latents.dtype)
                )

        return latents

    @torch.inference_mode()
    def step(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        conditioning_data: TextConditioningData,
        step_index: int,
        total_step_count: int,
        scheduler_step_kwargs: dict[str, Any],
        mask_guidance: AddsMaskGuidance | None,
        mask: torch.Tensor | None,
        masked_latents: torch.Tensor | None,
        control_data: list[ControlNetData] | None = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
    ):
        # invokeai_diffuser has batched timesteps, but diffusers schedulers expect a single value
        timestep = t[0]

        # Handle masked image-to-image (a.k.a inpainting).
        if mask_guidance is not None:
            # NOTE: This is intentionally done *before* self.scheduler.scale_model_input(...).
            latents = mask_guidance(latents, timestep)

        # TODO: should this scaling happen here or inside self._unet_forward?
        #     i.e. before or after passing it to InvokeAIDiffuserComponent
        latent_model_input = self.scheduler.scale_model_input(latents, timestep)

        # Handle ControlNet(s)
        down_block_additional_residuals = None
        mid_block_additional_residual = None
        if control_data is not None:
            down_block_additional_residuals, mid_block_additional_residual = self.invokeai_diffuser.do_controlnet_step(
                control_data=control_data,
                sample=latent_model_input,
                timestep=timestep,
                step_index=step_index,
                total_step_count=total_step_count,
                conditioning_data=conditioning_data,
            )

        # Handle T2I-Adapter(s)
        down_intrablock_additional_residuals = None
        if t2i_adapter_data is not None:
            accum_adapter_state = None
            for single_t2i_adapter_data in t2i_adapter_data:
                # Determine the T2I-Adapter weights for the current denoising step.
                first_t2i_adapter_step = math.floor(single_t2i_adapter_data.begin_step_percent * total_step_count)
                last_t2i_adapter_step = math.ceil(single_t2i_adapter_data.end_step_percent * total_step_count)
                t2i_adapter_weight = (
                    single_t2i_adapter_data.weight[step_index]
                    if isinstance(single_t2i_adapter_data.weight, list)
                    else single_t2i_adapter_data.weight
                )
                if step_index < first_t2i_adapter_step or step_index > last_t2i_adapter_step:
                    # If the current step is outside of the T2I-Adapter's begin/end step range, then set its weight to 0
                    # so it has no effect.
                    t2i_adapter_weight = 0.0

                # Apply the t2i_adapter_weight, and accumulate.
                if accum_adapter_state is None:
                    # Handle the first T2I-Adapter.
                    accum_adapter_state = [val * t2i_adapter_weight for val in single_t2i_adapter_data.adapter_state]
                else:
                    # Add to the previous adapter states.
                    for idx, value in enumerate(single_t2i_adapter_data.adapter_state):
                        accum_adapter_state[idx] += value * t2i_adapter_weight

            down_intrablock_additional_residuals = accum_adapter_state

        # Handle inpainting models.
        if is_inpainting_model(self.unet):
            # NOTE: These calls to add_inpainting_channels_to_latents(...) are intentionally done *after*
            # self.scheduler.scale_model_input(...) so that the scaling is not applied to the mask or reference image
            # latents.
            if mask is not None:
                if masked_latents is None:
                    raise ValueError("Source image required for inpaint mask when inpaint model used!")
                latent_model_input = self.add_inpainting_channels_to_latents(
                    latents=latent_model_input, masked_ref_image_latents=masked_latents, inpainting_mask=mask
                )
            else:
                # We are using an inpainting model, but no mask was provided, so we are not really "inpainting".
                # We generate a global mask and empty original image so that we can still generate in this
                # configuration.
                # TODO(ryand): Should we just raise an exception here instead? I can't think of a use case for wanting
                # to do this.
                # TODO(ryand): If we decide that there is a good reason to keep this, then we should generate the 'fake'
                # mask and original image once rather than on every denoising step.
                latent_model_input = self.add_inpainting_channels_to_latents(
                    latents=latent_model_input,
                    masked_ref_image_latents=torch.zeros_like(latent_model_input[:1]),
                    inpainting_mask=torch.ones_like(latent_model_input[:1, :1]),
                )

        uc_noise_pred, c_noise_pred = self.invokeai_diffuser.do_unet_step(
            sample=latent_model_input,
            timestep=t,  # TODO: debug how handled batched and non batched timesteps
            step_index=step_index,
            total_step_count=total_step_count,
            conditioning_data=conditioning_data,
            ip_adapter_data=ip_adapter_data,
            down_block_additional_residuals=down_block_additional_residuals,  # for ControlNet
            mid_block_additional_residual=mid_block_additional_residual,  # for ControlNet
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,  # for T2I-Adapter
        )

        guidance_scale = conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[step_index]

        noise_pred = self.invokeai_diffuser._combine(uc_noise_pred, c_noise_pred, guidance_scale)
        guidance_rescale_multiplier = conditioning_data.guidance_rescale_multiplier
        if guidance_rescale_multiplier > 0:
            noise_pred = self._rescale_cfg(
                noise_pred,
                c_noise_pred,
                guidance_rescale_multiplier,
            )

        # compute the previous noisy sample x_t -> x_t-1
        step_output = self.scheduler.step(noise_pred, timestep, latents, **scheduler_step_kwargs)

        # TODO: discuss injection point options. For now this is a patch to get progress images working with inpainting
        # again.
        if mask_guidance is not None:
            # Apply the mask to any "denoised" or "pred_original_sample" fields.
            if hasattr(step_output, "denoised"):
                step_output.pred_original_sample = mask_guidance(step_output.denoised, self.scheduler.timesteps[-1])
            elif hasattr(step_output, "pred_original_sample"):
                step_output.pred_original_sample = mask_guidance(
                    step_output.pred_original_sample, self.scheduler.timesteps[-1]
                )
            else:
                step_output.pred_original_sample = mask_guidance(latents, self.scheduler.timesteps[-1])

        return step_output

    @staticmethod
    def _rescale_cfg(total_noise_pred, pos_noise_pred, multiplier=0.7):
        """Implementation of Algorithm 2 from https://arxiv.org/pdf/2305.08891.pdf."""
        ro_pos = torch.std(pos_noise_pred, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(total_noise_pred, dim=(1, 2, 3), keepdim=True)

        x_rescaled = total_noise_pred * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * total_noise_pred
        return x_final

    def _unet_forward(
        self,
        latents,
        t,
        text_embeddings,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """predict the noise residual"""
        # First three args should be positional, not keywords, so torch hooks can see them.
        return self.unet(
            latents,
            t,
            text_embeddings,
            cross_attention_kwargs=cross_attention_kwargs,
            **kwargs,
        ).sample

from abc import ABC
import dataclasses
from functools import partial
from tqdm.auto import tqdm

@dataclass
class InjectionInfo:
    type: str
    name: str
    order: str

def modifier(name, order="any"):
    def _decorator(func):
        func.__inj_info__ = InjectionInfo(
            type="modifier",
            name=name,
            order=order,
        )
        return func
    return _decorator

def override(name):
    def _decorator(func):
        func.__inj_info__ = InjectionInfo(
            type="override",
            name=name,
            order="any",
        )
        return func
    return _decorator

class ExtensionBase:
    def __init__(self, priority):
        self.priority = priority
        self.injections = []
        for func_name in dir(self):
            func = getattr(self, func_name)
            if not callable(func) or not hasattr(func, "__inj_info__"):
                continue

            self.injections.append((func.__inj_info__, func))


class InpaintExt(ExtensionBase):
    mask: torch.Tensor
    masked_latents: Optional[torch.Tensor]
    is_gradient_mask: bool

    def __init__(self, mask, masked_latents, is_gradient_mask, priority):
        super().__init__(priority=priority)
        self.mask = mask
        self.masked_latents = masked_latents
        self.is_gradient_mask = is_gradient_mask

    def _is_inpaint_model(self, unet: UNet2DConditionModel):
        return unet.conv_in.in_channels == 9

    def _apply_mask(self, ctx, latents: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size = latents.size(0)
        mask = einops.repeat(self.mask, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if t.dim() == 0:
            # some schedulers expect t to be one-dimensional.
            # TODO: file diffusers bug about inconsistency?
            t = einops.repeat(t, "-> batch", batch=batch_size)
        # Noise shouldn't be re-randomized between steps here. The multistep schedulers
        # get very confused about what is happening from step to step when we do that.
        mask_latents = ctx.scheduler.add_noise(ctx.orig_latents, self.noise, t)
        # TODO: Do we need to also apply scheduler.scale_model_input? Or is add_noise appropriately scaled already?
        # mask_latents = self.scheduler.scale_model_input(mask_latents, t)
        mask_latents = einops.repeat(mask_latents, "b c h w -> (repeat b) c h w", repeat=batch_size)
        if self.is_gradient_mask:
            threshhold = (t.item()) / ctx.scheduler.config.num_train_timesteps
            mask_bool = mask > threshhold  # I don't know when mask got inverted, but it did
            masked_input = torch.where(mask_bool, latents, mask_latents)
        else:
            masked_input = torch.lerp(mask_latents.to(dtype=latents.dtype), latents, mask.to(dtype=latents.dtype))
        return masked_input


    @modifier("pre_denoise_loop")
    def init_tensors(self, ctx):
        if self._is_inpaint_model(ctx.unet):
            if self.mask is None:
                self.mask = torch.ones_like(ctx.latents[:1, :1])
            self.mask.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

            if self.masked_latents is None:
                self.masked_latents = torch.zeros_like(ctx.latents[:1])
            self.masked_latents.to(device=ctx.latents.device, dtype=ctx.latents.dtype)

        else:
            #self.orig_latents = ctx.orig_latents
            self.noise = ctx.noise
            if self.noise is None:
                self.noise = torch.randn(
                    ctx.orig_latents.shape,
                    dtype=torch.float32,
                    device="cpu",
                    generator=torch.Generator(device="cpu").manual_seed(ctx.seed),
                ).to(device=ctx.orig_latents.device, dtype=ctx.orig_latents.dtype)

    @modifier("pre_step") # last?
    def apply_mask_to_latents(self, ctx):
        if self._is_inpaint_model(ctx.unet) or self.mask is None:
            return
        ctx.latents = self._apply_mask(ctx, ctx.latents, ctx.timestep)

    @modifier("pre_unet_forward", order="last")
    def append_inpaint_layers(self, ctx):
        if not self._is_inpaint_model(ctx.unet):
            return

        batch_size = ctx.unet_kwargs.sample.shape[0]
        b_mask = torch.cat([self.mask] * batch_size)
        b_masked_latents = torch.cat([self.masked_latents] * batch_size)
        ctx.unet_kwargs.sample = torch.cat(
            [ctx.unet_kwargs.sample, b_mask, b_masked_latents],
            dim=1,
        )

    @modifier("post_step", order="first")
    def apply_mask_to_preview(self, ctx):
        if self._is_inpaint_model(ctx.unet) or self.mask is None:
            return

        timestep = ctx.scheduler.timesteps[-1]
        if hasattr(ctx.step_output, "denoised"):
            ctx.step_output.denoised = self._apply_mask(ctx, ctx.step_output.denoised, timestep)
        elif hasattr(ctx.step_output, "pred_original_sample"):
            ctx.step_output.pred_original_sample = self._apply_mask(ctx, ctx.step_output.pred_original_sample, timestep)
        else:
            ctx.step_output.pred_original_sample = self._apply_mask(ctx, ctx.step_output.prev_sample, timestep)

    @modifier("post_denoise_loop") # last?
    def restore_unmasked(self, ctx):
        if self.mask is None:
            return

        # restore unmasked part after the last step is completed
        # in-process masking happens before each step
        if self.is_gradient_mask:
            ctx.latents = torch.where(self.mask > 0, ctx.latents, ctx.orig_latents)
        else:
            ctx.latents = torch.lerp(
                ctx.orig_latents,
                ctx.latents.to(dtype=ctx.orig_latents.dtype),
                self.mask.to(dtype=ctx.orig_latents.dtype),
            )


class PreviewExt(ExtensionBase):
    def __init__(self, callback, priority):
        super().__init__(priority=priority)
        self.callback = callback

    @modifier("pre_denoise_loop")
    def initial_preview(self, ctx, order="last"):
        self.callback(
            PipelineIntermediateState(
                step=-1,
                order=ctx.scheduler.order,
                total_steps=len(ctx.timesteps),
                timestep=int(ctx.scheduler.config.num_train_timesteps), # TODO: is there any code which uses it?
                latents=ctx.latents,
            )
        )

    @modifier("post_step", order="last")
    def step_preview(self, ctx):
        if hasattr(ctx.step_output, "denoised"):
            predicted_original = ctx.step_output.denoised
        elif hasattr(ctx.step_output, "pred_original_sample"):
            predicted_original = ctx.step_output.pred_original_sample
        else:
            predicted_original = ctx.step_output.prev_sample


        self.callback(
            PipelineIntermediateState(
                step=ctx.step_index,
                order=ctx.scheduler.order,
                total_steps=len(ctx.timesteps),
                timestep=int(ctx.timestep), # TODO: is there any code which uses it?
                latents=ctx.step_output.prev_sample,
                predicted_original=predicted_original, # TODO: is there any reason for additional field?
            )
        )


class RescaleCFGExt(ExtensionBase):
    def __init__(self, guidance_rescale_multiplier, priority):
        super().__init__(priority=priority)
        self.guidance_rescale_multiplier = guidance_rescale_multiplier

    @staticmethod
    def _rescale_cfg(total_noise_pred, pos_noise_pred, multiplier=0.7):
        """Implementation of Algorithm 2 from https://arxiv.org/pdf/2305.08891.pdf."""
        ro_pos = torch.std(pos_noise_pred, dim=(1, 2, 3), keepdim=True)
        ro_cfg = torch.std(total_noise_pred, dim=(1, 2, 3), keepdim=True)

        x_rescaled = total_noise_pred * (ro_pos / ro_cfg)
        x_final = multiplier * x_rescaled + (1.0 - multiplier) * total_noise_pred
        return x_final

    @modifier("modify_noise_prediction")
    def rescale_noise_pred(self, ctx):
        if self.guidance_rescale_multiplier > 0:
            ctx.noise_pred = self._rescale_cfg(
                ctx.noise_pred,
                ctx.positive_noise_pred,
                self.guidance_rescale_multiplier,
            )


class ExtModifiersApi(ABC):
    def pre_denoise_loop(self, ctx):
        pass

    def post_denoise_loop(self, ctx):
        pass

    def pre_step(self, ctx):
        pass

    def post_step(self, ctx):
        pass

    def modify_noise_prediction(self, ctx):
        pass

    def pre_unet_forward(self, ctx):
        pass

class ExtOverridesApi(ABC):
    def step(self, orig_func, ctx):
        pass

    def combine_noise(self, orig_func, ctx):
        pass

class ProxyCallsClass:
    def __init__(self, handler):
        self._handler = handler
    def __getattr__(self, item):
        return partial(self._handler, item)

class InjectionPoint:
    def __init__(self, name):
        self.name = name
        self.first = []
        self.any = []
        self.last = []

    def add(self, func, order):
        if order == "first":
            self.first.append(func)
        elif order == "last":
            self.last.append(func)
        else: # elif order == "any":
            self.any.append(func)

    def __call__(self, *args, **kwargs):
        for func in self.first:
            func(*args, **kwargs)
        for func in self.any:
            func(*args, **kwargs)
        for func in reversed(self.last):
            func(*args, **kwargs)


class ExtensionsManager:
    def __init__(self):
        self.extensions = []
        self.ordered_extensions = []
        self.injections = dict(
            modifiers=dict(),
            overrides=dict(),
        )

        self.modifiers: ExtModifiersApi = ProxyCallsClass(self.call_modifier)
        self.overrides: ExtOverridesApi = ProxyCallsClass(self.call_override)


    def add_extension(self, ext):
        self.extensions.append(ext)
        self.ordered_extensions = sorted(self.extensions, reverse=True, key=lambda ext: ext.priority)

        self.injections = dict(
            modifier=dict(),
            override=dict(),
        )

        for ext in self.ordered_extensions:
            for inj_info, inj_func in ext.injections:
                if inj_info.type == "modifier":
                    if inj_info.name not in self.injections[inj_info.type]:
                        self.injections[inj_info.type][inj_info.name] = InjectionPoint(inj_info.name)
                    self.injections[inj_info.type][inj_info.name].add(inj_func, inj_info.order)
                else:
                    if inj_info.name in self.injections[inj_info.type]:
                        raise Exception(f"Already overloaded - {inj_info.name}")
                    self.injections[inj_info.type] = inj_func

    def call_modifier(self, name, ctx):
        if name in self.injections["modifier"]:
            self.injections["modifier"][name](ctx)

    def call_override(self, name, orig_func, ctx):
        if name in self.injections["override"]:
            return self.injections["override"][name](orig_func, ctx)
        else:
            return orig_func(ctx, self)

@dataclass
class UNetKwargs:
    sample: torch.Tensor
    timestep: Union[torch.Tensor, float, int]
    encoder_hidden_states: torch.Tensor

    class_labels: Optional[torch.Tensor] = None
    timestep_cond: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None
    mid_block_additional_residual: Optional[torch.Tensor] = None
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None
    encoder_attention_mask: Optional[torch.Tensor] = None
    #return_dict: bool = True

@dataclass
class DenoiseContext:
    latents: torch.Tensor
    scheduler_step_kwargs: dict[str, Any]
    conditioning_data: TextConditioningData
    noise: Optional[torch.Tensor]
    seed: int
    timesteps: torch.Tensor
    init_timestep: torch.Tensor
    
    unet: UNet2DConditionModel
    scheduler: SchedulerMixin

    orig_latents: Optional[torch.Tensor] = None
    unet_kwargs: Optional[UNetKwargs] = None

    latent_model_input: Optional[torch.Tensor] = None
    negative_noise_pred: Optional[torch.Tensor] = None
    positive_noise_pred: Optional[torch.Tensor] = None
    noise_pred: Optional[torch.Tensor] = None

    def __delattr__(self, name: str):
        setattr(self, name, None)

    @classmethod
    def from_kwargs(cls, **kwargs):
        names = set([f.name for f in dataclasses.fields(cls)])
        return cls(**{
            k: v for k, v in kwargs.items() 
            #if k in inspect.signature(cls).parameters
            if k in names
        })


class StableDiffusionBackend:
    def __init__(self, unet, scheduler):
        self.unet = unet
        self.scheduler = scheduler
        self.sequential_guidance = False

    def latents_from_embeddings(
        self,
        latents: torch.Tensor,
        scheduler_step_kwargs: dict[str, Any],
        conditioning_data: TextConditioningData,
        noise: Optional[torch.Tensor],
        seed: int,
        timesteps: torch.Tensor,
        init_timestep: torch.Tensor,
        callback: Callable[[PipelineIntermediateState], None],
        control_data: list[ControlNetData] | None = None,
        ip_adapter_data: Optional[list[IPAdapterData]] = None,
        t2i_adapter_data: Optional[list[T2IAdapterData]] = None,
        mask: Optional[torch.Tensor] = None,
        masked_latents: Optional[torch.Tensor] = None,
        is_gradient_mask: bool = False,
    ) -> torch.Tensor:
        ctx = DenoiseContext.from_kwargs(**locals(), unet=self.unet, scheduler=self.scheduler)
        ext_controller = ExtensionsManager()
        if mask is not None or is_inpainting_model(ctx.unet):
            ext_controller.add_extension(InpaintExt(mask, masked_latents, is_gradient_mask, priority=200))

        #if t2i_adapter_data is not None:
        #    for t2i_adapter in t2i_adapter_data:
        #        ext_controller.add_extension(T2IAdapterExt(t2i_adapter, priority=100))

        #if ip_adapter_data is not None:
        #    for ip_adapter in ip_adapter_data:
        #        ext_controller.add_extension(IPAdapterExt(ip_adapter, priority=100))

        #if control_data is not None:
        #    for controlnet in control_data:
        #        ext_controller.add_extension(ControlNetExt(controlnet, priority=100))

        ext_controller.add_extension(PreviewExt(callback, priority=99999))

        if ctx.conditioning_data.guidance_rescale_multiplier > 0:
            ext_controller.add_extension(RescaleCFGExt(ctx.conditioning_data.guidance_rescale_multiplier, priority=100))

        return self.new_gen(ctx, ext_controller)




    def new_gen(self, ctx, ext_controller):
        if ctx.init_timestep.shape[0] == 0:
            return ctx.latents

        ctx.orig_latents = ctx.latents.clone()

        if ctx.noise is not None:
            batch_size = ctx.latents.shape[0]
            # latents = noise * self.scheduler.init_noise_sigma # it's like in t2l according to diffusers
            ctx.latents = ctx.scheduler.add_noise(ctx.latents, ctx.noise, ctx.init_timestep.expand(batch_size))

        # if no work to do, return latents
        if ctx.timesteps.shape[0] == 0:
            return ctx.latents

        # ext: inpaint[pre_denoise_loop, priority=normal] (maybe init, but not sure if it needed)
        # ext: preview[pre_denoise_loop, priority=low]
        ext_controller.modifiers.pre_denoise_loop(ctx)

        # patch on nodes level
        # apply attention to unet and call in all extensions .apply_attention_processor(CustomAttentionProcessor)
        # ip adapters - for now add .add_ip_adapter method in CustomAttentionProcessor, in patcher iterate through unet processors and call this method
        #with UNetAttentionPatcher_new(ctx.unet, addons):
        for ctx.step_index, ctx.timestep in enumerate(tqdm(ctx.timesteps)):

            # ext: inpaint (apply mask to latents on non-inpaint models)
            ext_controller.modifiers.pre_step(ctx)

            # ext: tiles? [override: step]
            ctx.step_output = ext_controller.overrides.step(self.step, ctx) # , ext_controller)

            # ext: inpaint[post_step, priority=high] (apply mask to preview on non-inpaint models)
            # ext: preview[post_step, priority=low]
            ext_controller.modifiers.post_step(ctx)

            ctx.latents = ctx.step_output.prev_sample

        # ext: inpaint[post_denoise_loop] (restore unmasked part)
        ext_controller.modifiers.post_denoise_loop(ctx)
        return ctx.latents


    @torch.inference_mode()
    def step(self, ctx, ext_controller):
        ctx.latent_model_input = ctx.scheduler.scale_model_input(ctx.latents, ctx.timestep)

        if self.sequential_guidance:
            conditioning_call = self._apply_standard_conditioning_sequentially
        else:
            conditioning_call = self._apply_standard_conditioning

        # not sure if here needed override
        ctx.negative_noise_pred, ctx.positive_noise_pred = conditioning_call(ctx, ext_controller)

        # ext: override combine_noise
        ctx.noise_pred = ext_controller.overrides.combine_noise(self.combine_noise, ctx)

        # ext: cfg_rescale [modify_noise_prediction]
        ext_controller.modifiers.modify_noise_prediction(ctx)

        # compute the previous noisy sample x_t -> x_t-1
        step_output = ctx.scheduler.step(ctx.noise_pred, ctx.timestep, ctx.latents, **ctx.scheduler_step_kwargs)

        # del locals
        del ctx.latent_model_input
        del ctx.negative_noise_pred
        del ctx.positive_noise_pred
        del ctx.noise_pred

        return step_output

    @staticmethod
    def combine_noise(ctx, ext_controller):
        guidance_scale = ctx.conditioning_data.guidance_scale
        if isinstance(guidance_scale, list):
            guidance_scale = guidance_scale[ctx.step_index]

        return torch.lerp(ctx.negative_noise_pred, ctx.positive_noise_pred, guidance_scale)
        #return ctx.negative_noise_pred + guidance_scale * (ctx.positive_noise_pred - ctx.negative_noise_pred)

    def _apply_standard_conditioning(self, ctx, ext_controller) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the conditioned and unconditioned UNet forward passes in a single batch for faster inference speed at
        the cost of higher memory usage.
        """

        ctx.unet_kwargs = UNetKwargs(
            sample=torch.cat([ctx.latent_model_input] * 2),
            timestep=ctx.timestep,
            encoder_hidden_states=None, # set later by conditoning

            cross_attention_kwargs=dict(
                percent_through=ctx.step_index / len(ctx.timesteps), # ctx.total_steps,
            )
        )
        ctx.conditioning_data.to_unet_kwargs(ctx.unet_kwargs, "both")

        # ext: controlnet/ip/t2i [pre_unet_forward]
        ext_controller.modifiers.pre_unet_forward(ctx)

        # ext: inpaint [pre_unet_forward, priority=low]
        # or
        # ext: inpaint [override: unet_forward]
        both_results = self._unet_forward(
            **vars(ctx.unet_kwargs),
            #torch.cat([ctx.latent_model_input] * 2),
            #ctx.timestep,
            #**unet_kwargs,
        )
        negative_next_x, positive_next_x = both_results.chunk(2)
        # del locals
        del ctx.unet_kwargs
        return negative_next_x, positive_next_x


    def _apply_standard_conditioning_sequentially(self, ctx, ext_controller):
        """Runs the conditioned and unconditioned UNet forward passes sequentially for lower memory usage at the cost of
        slower execution speed.
        """

        ###################
        # Negative pass
        ###################

        ctx.unet_kwargs = UNetKwargs(
            sample=ctx.latent_model_input,
            timestep=ctx.timestep,
            encoder_hidden_states=None, # set later by conditoning

            cross_attention_kwargs=dict(
                percent_through=ctx.step_index / len(ctx.timesteps), # ctx.total_steps,
            )
        )
        ctx.conditioning_data.to_unet_kwargs(ctx.unet_kwargs, "negative")

        # ext: controlnet/ip/t2i [pre_unet_forward]
        ext_controller.modifiers.pre_unet_forward(ctx)

        # ext: inpaint [pre_unet_forward, priority=low]
        # or
        # ext: inpaint [override: unet_forward]
        negative_next_x = self._unet_forward(
            **vars(ctx.unet_kwargs),
            #ctx.latent_model_input,
            #ctx.timestep,
            #**ctx.unet_kwargs,
        )

        del ctx.unet_kwargs
        # TODO: gc.collect() ?

        ###################
        # Positive pass
        ###################

        ctx.unet_kwargs = UNetKwargs(
            sample=ctx.latent_model_input,
            timestep=ctx.timestep,
            encoder_hidden_states=None, # set later by conditoning

            cross_attention_kwargs=dict(
                percent_through=ctx.step_index / len(ctx.timesteps), # ctx.total_steps,
            )
        )
        ctx.conditioning_data.to_unet_kwargs(ctx.unet_kwargs, "positive")

        # ext: controlnet/ip/t2i [pre_unet_forward]
        ext_controller.modifiers.pre_unet_forward(ctx)

        # ext: inpaint [pre_unet_forward, priority=low]
        # or
        # ext: inpaint [override: unet_forward]
        positive_next_x = self._unet_forward(
            **vars(ctx.unet_kwargs),
            #ctx.latent_model_input,
            #ctx.timestep,
            #**ctx.unet_kwargs,
        )

        del ctx.unet_kwargs
        # TODO: gc.collect() ?

        return negative_next_x, positive_next_x


    def _unet_forward(self, **kwargs):
        return self.unet(**kwargs).sample

