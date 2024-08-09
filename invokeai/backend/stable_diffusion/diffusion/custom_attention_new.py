from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from dataclasses import dataclass
import psutil
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

from invokeai.app.services.config.config_default import get_config
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.stable_diffusion.extensions_manager import CallbackApi

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.stable_diffusion.diffusion.regional_prompt_data import RegionalPromptData

@dataclass
class AttentionContext:
    processor: CustomAttentionProcessor
    is_cross_attention: bool

    attn: Attention
    hidden_states: torch.Tensor
    encoder_hidden_states: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    temb: Optional[torch.Tensor]
    residual: torch.Tensor
    input_ndim: int

    batch_size: Optional[int] = None
    key_length: Optional[int] = None
    query_length: Optional[int] = None

    # in sd always none(as I know)
    channel: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None

    query: Optional[torch.Tensor] = None
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None


class CustomAttnProcessorNew:
    """A custom implementation of attention processor that supports additional Invoke features.
    This implementation is based on
    SlicedAttnProcessor (https://github.com/huggingface/diffusers/blob/fcfa270fbd1dc294e2f3a505bae6bcb791d721c3/src/diffusers/models/attention_processor.py#L1616)
    AttnProcessor2_0 (https://github.com/huggingface/diffusers/blob/fcfa270fbd1dc294e2f3a505bae6bcb791d721c3/src/diffusers/models/attention_processor.py#L1204)
    """

    def __init__(self, attention_id: Optional[int] = None, attention_key: Optional[str] = None):
        assert (attention_id is None) == (attention_key is None)
        self.attention_id = attention_id
        self.attention_key = attention_key

        config = get_config()
        self._attention_type = config.attention_type
        if self._attention_type == "auto":
            self._attention_type = self._select_attention_type()

        self._slice_size = config.attention_slice_size
        if self._slice_size == "auto":
            self._slice_size = self._select_slice_size()

        if self._attention_type == "xformers" and xformers is None:
            raise ImportError("xformers attention requires xformers module to be installed.")

    def _select_attention_type(self) -> str:
        device = TorchDevice.choose_torch_device()
        # On some mps system normal attention still faster than torch-sdp, on others - on par
        # Results torch-sdp vs normal attention
        # gogurt: 67.993s vs 67.729s
        # Adreitz: 260.868s vs 226.638s
        if device.type == "mps":
            return "normal"
        elif device.type == "cuda":
            # Flash Attention is supported from sm80 compute capability onwards in PyTorch
            # https://pytorch.org/blog/accelerated-pytorch-2/
            if torch.cuda.get_device_capability("cuda")[0] < 8 and xformers is not None:
                return "xformers"
            else:
                return "torch-sdp"
        else:  # cpu
            return "torch-sdp"

    def _select_slice_size(self) -> str:
        device = TorchDevice.choose_torch_device()
        if device.type in ["cpu", "mps"]:
            total_ram_gb = math.ceil(psutil.virtual_memory().total / 2**30)
            if total_ram_gb <= 16:
                return "max"
            if total_ram_gb <= 32:
                return "balanced"
            return "none"
        elif device.type == "cuda":
            total_vram_gb = math.ceil(torch.cuda.get_device_properties(device).total_memory / 2**30)
            if total_vram_gb <= 4:
                return "max"
            if total_vram_gb <= 6:
                return "balanced"
            return "none"
        else:
            raise ValueError(f"Unknown device: {device.type}")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        # modular backend fields
        denoise_ctx: Optional[DenoiseContext] = None,
        # For Regional Prompting:
        regional_prompt_data: Optional[RegionalPromptData] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if denoise_ctx is not None:
            assert self.attention_key is not None and self.attention_id is not None

        ctx = AttentionContext(
            processor=self,
            is_cross_attention=encoder_hidden_states is not None,
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            residual=hidden_states,
            input_ndim=hidden_states.ndim,
        )

        #ctx.residual = ctx.hidden_states

        if ctx.attn.spatial_norm is not None:
            ctx.hidden_states = ctx.attn.spatial_norm(ctx.hidden_states, ctx.temb)

        #ctx.input_ndim = ctx.hidden_states.ndim

        if ctx.input_ndim == 4:
            ctx.batch_size, ctx.channel, ctx.height, ctx.width = ctx.hidden_states.shape
            ctx.hidden_states = ctx.hidden_states.view(ctx.batch_size, ctx.channel, ctx.height * ctx.width).transpose(1, 2)

        ctx.batch_size, ctx.key_length, _ = (
            ctx.hidden_states.shape if ctx.encoder_hidden_states is None else ctx.encoder_hidden_states.shape
        )
        ctx.query_length = ctx.hidden_states.shape[1]

        # TODO: try redo as extension after conditioning rewrite(attention couple)
        if regional_prompt_data is not None and ctx.is_cross_attention:
            prompt_region_attention_mask = regional_prompt_data.get_cross_attn_mask(
                query_seq_len=ctx.query_length, key_seq_len=ctx.key_length
            )

            if ctx.attention_mask is None:
                ctx.attention_mask = prompt_region_attention_mask
            else:
                ctx.attention_mask += prompt_region_attention_mask

        # ext: regional prompts(attention couple)
        if denoise_ctx is not None:
            denoise_ctx.ext_manager.callback.pre_prepare_attention_mask(ctx, denoise_ctx)
        ctx.attention_mask = ctx.attn.prepare_attention_mask(ctx.attention_mask, ctx.key_length, ctx.batch_size)
        if denoise_ctx is not None:
            denoise_ctx.ext_manager.callback.post_prepare_attention_mask(ctx, denoise_ctx)

        if ctx.attn.group_norm is not None:
            ctx.hidden_states = ctx.attn.group_norm(ctx.hidden_states.transpose(1, 2)).transpose(1, 2)

        ctx.query = ctx.attn.to_q(ctx.hidden_states)

        if ctx.encoder_hidden_states is None:
            ctx.encoder_hidden_states = ctx.hidden_states
        elif ctx.attn.norm_cross:
            ctx.encoder_hidden_states = ctx.attn.norm_encoder_hidden_states(ctx.encoder_hidden_states)

        ctx.key = ctx.attn.to_k(ctx.encoder_hidden_states)
        ctx.value = ctx.attn.to_v(ctx.encoder_hidden_states)

        if denoise_ctx is not None:
            denoise_ctx.ext_manager.callback.pre_run_attention(ctx, denoise_ctx)
        ctx.hidden_states = self.run_attention(
            attn=ctx.attn,
            query=ctx.query,
            key=ctx.key,
            value=ctx.value,
            attention_mask=ctx.attention_mask,
        )
        # ext: ip adapter
        if denoise_ctx is not None:
            denoise_ctx.ext_manager.callback.post_run_attention(ctx, denoise_ctx)

        # linear proj
        ctx.hidden_states = ctx.attn.to_out[0](ctx.hidden_states)
        # dropout
        ctx.hidden_states = ctx.attn.to_out[1](ctx.hidden_states)

        if ctx.input_ndim == 4:
            ctx.hidden_states = ctx.hidden_states.transpose(-1, -2).reshape(ctx.batch_size, ctx.channel, ctx.height, ctx.width)

        if ctx.attn.residual_connection:
            ctx.hidden_states += ctx.residual

        ctx.hidden_states /= ctx.attn.rescale_output_factor
        return ctx.hidden_states

    def _get_slice_size(self, attn: Attention) -> Optional[int]:
        if self._slice_size == "none":
            return None
        if isinstance(self._slice_size, int):
            return self._slice_size

        if self._slice_size == "max":
            return 1
        if self._slice_size == "balanced":
            return max(1, attn.sliceable_head_dim // 2)

        raise ValueError(f"Incorrect slice_size value: {self.slice_size}")

    def run_attention(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        slice_size = self._get_slice_size(attn)
        if slice_size is not None:
            return self._run_attention_sliced(
                attn=attn,
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                slice_size=slice_size,
            )

        if self._attention_type == "torch-sdp":
            attn_call = self._run_attention_sdp
        elif self._attention_type == "normal":
            attn_call = self._run_attention_normal
        elif self._attention_type == "xformers":
            attn_call = self._run_attention_xformers
        else:
            raise Exception(f"Unknown attention type: {self._attention_type}")

        return attn_call(
            attn=attn,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

    def _run_attention_normal(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states

    def _run_attention_xformers(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if attention_mask is not None:
            # expand our mask's singleton query_length dimension:
            #   [batch*heads,            1, key_length] ->
            #   [batch*heads, query_length, key_length]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_length, key_length]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            attention_mask = attention_mask.expand(-1, query.shape[1], -1)

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=None, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states

    def _run_attention_sdp(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = key.shape[0]
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=attn.scale
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states

    def _run_attention_sliced(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        slice_size: int,
    ) -> torch.Tensor:
        dim = query.shape[-1]

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads), device=query.device, dtype=query.dtype
        )

        for i in range((batch_size_attention - 1) // slice_size + 1):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            value_slice = value[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            if self._attention_type == "normal":
                attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                torch.bmm(attn_slice, value_slice, out=hidden_states[start_idx:end_idx])
                del attn_slice
            elif self._attention_type == "xformers":
                if attn_mask_slice is not None:
                    attn_mask_slice = attn_mask_slice.expand(-1, query.shape[1], -1)

                hidden_states[start_idx:end_idx] = xformers.ops.memory_efficient_attention(
                    query_slice, key_slice, value_slice, attn_bias=attn_mask_slice, op=None, scale=attn.scale
                )
            elif self._attention_type == "torch-sdp":
                if attn_mask_slice is not None:
                    attn_mask_slice = attn_mask_slice.unsqueeze(0)

                hidden_states[start_idx:end_idx] = F.scaled_dot_product_attention(
                    query_slice.unsqueeze(0),
                    key_slice.unsqueeze(0),
                    value_slice.unsqueeze(0),
                    attn_mask=attn_mask_slice,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=attn.scale,
                ).squeeze(0)
            else:
                raise ValueError(f"Unknown attention type: {self._attention_type}")

        hidden_states = attn.batch_to_head_dim(hidden_states)
        return hidden_states
