from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

import invokeai.backend.util.logging as logger
from invokeai.app.services.config.config_default import get_config
from invokeai.backend.ip_adapter.ip_attention_weights import IPAttentionProcessorWeights
from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
from invokeai.backend.stable_diffusion.diffusion.regional_prompt_data import RegionalPromptData
from invokeai.backend.util.devices import TorchDevice


@dataclass
class IPAdapterAttentionWeights:
    ip_adapter_weights: IPAttentionProcessorWeights
    skip: bool


class CustomAttnProcessor:
    """A custom implementation of attention processor that supports additional Invoke features.
    This implementation is based on
    SlicedAttnProcessor (https://github.com/huggingface/diffusers/blob/fcfa270fbd1dc294e2f3a505bae6bcb791d721c3/src/diffusers/models/attention_processor.py#L1616)
    AttnProcessor2_0 (https://github.com/huggingface/diffusers/blob/fcfa270fbd1dc294e2f3a505bae6bcb791d721c3/src/diffusers/models/attention_processor.py#L1204)
    Supported custom features:
    - IP-Adapter
    - Regional prompt attention
    """

    def __init__(
        self,
        ip_adapter_attention_weights: Optional[List[IPAdapterAttentionWeights]] = None,
    ):
        """Initialize a CustomAttnProcessor.
        Note: Arguments that are the same for all attention layers are passed to __call__(). Arguments that are
        layer-specific are passed to __init__().
        Args:
            ip_adapter_weights: The IP-Adapter attention weights. ip_adapter_weights[i] contains the attention weights
                for the i'th IP-Adapter.
        """

        self._ip_adapter_attention_weights = ip_adapter_attention_weights
        self.attention_type, self.slice_size = self._select_attention()

    def _select_attention(self):
        config = get_config()
        attention_type = config.attention_type
        if attention_type in ["normal", "xformers"]:
            logger.warning(f'Attention "{attention_type}" no longer supported, "torch-sdp" will be used instead.')
            attention_type = "torch-sdp"

        if attention_type == "auto":
            exec_device = TorchDevice.choose_torch_device()
            if exec_device.type == "mps":
                attention_type = "sliced"
            else:
                attention_type = "torch-sdp"

        if attention_type == "torch-sdp" and not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("torch-sdp attention requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        slice_size = None
        if attention_type == "sliced":
            slice_size = config.attention_slice_size
            if slice_size not in ["auto", "balanced", "max"] and not isinstance(slice_size, int):
                raise ValueError(f"Unsupported attention_slice_size: {slice_size}")
            if slice_size == "balanced":
                slice_size = "auto"

        return attention_type, slice_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        # For Regional Prompting:
        regional_prompt_data: Optional[RegionalPromptData] = None,
        percent_through: Optional[torch.Tensor] = None,
        # For IP-Adapter:
        regional_ip_data: Optional[RegionalIPData] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # If true, we are doing cross-attention, if false we are doing self-attention.
        is_cross_attention = encoder_hidden_states is not None

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        query_length = hidden_states.shape[1]

        # Regional Prompt Attention Mask
        if regional_prompt_data is not None and is_cross_attention:
            prompt_region_attention_mask = regional_prompt_data.get_cross_attn_mask(
                query_seq_len=query_length, key_seq_len=key_length
            )

            if attention_mask is None:
                attention_mask = prompt_region_attention_mask
            else:
                attention_mask = prompt_region_attention_mask + attention_mask

        attention_mask = attn.prepare_attention_mask(attention_mask, key_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        hidden_states = self.run_attention(
            attn=attn,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

        if is_cross_attention:
            hidden_states = self.run_ip_adapters(
                attn=attn,
                hidden_states=hidden_states,
                regional_ip_data=regional_ip_data,
                query_length=query_length,
                query=query,
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states

    def run_ip_adapters(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        regional_ip_data: Optional[RegionalIPData],
        query_length: int,  # TODO: just read from query?
        query: torch.Tensor,
    ) -> torch.Tensor:
        if self._ip_adapter_attention_weights is None:
            # If IP-Adapter is not enabled, then regional_ip_data should not be passed in.
            assert regional_ip_data is None
            return hidden_states

        assert regional_ip_data is not None
        ip_masks = regional_ip_data.get_masks(query_seq_len=query_length)

        assert (
            len(regional_ip_data.image_prompt_embeds)
            == len(self._ip_adapter_attention_weights)
            == len(regional_ip_data.scales)
            == ip_masks.shape[1]
        )

        for ipa_index, ip_hidden_states in enumerate(regional_ip_data.image_prompt_embeds):
            # The batch dimensions should match.
            # assert ip_hidden_states.shape[0] == encoder_hidden_states.shape[0]
            # The token_len dimensions should match.
            # assert ip_hidden_states.shape[-1] == encoder_hidden_states.shape[-1]

            if self._ip_adapter_attention_weights[ipa_index].skip:
                continue

            ipa_weights = self._ip_adapter_attention_weights[ipa_index].ip_adapter_weights
            ipa_scale = regional_ip_data.scales[ipa_index]
            ip_mask = ip_masks[0, ipa_index, ...]

            # Expected ip_hidden_state shape: (batch_size, num_ip_images, ip_seq_len, ip_image_embedding)
            ip_key = ipa_weights.to_k_ip(ip_hidden_states)
            ip_value = ipa_weights.to_v_ip(ip_hidden_states)

            ip_hidden_states = self.run_attention(
                attn=attn,
                query=query,
                key=ip_key,
                value=ip_value,
                attention_mask=None,
            )

            # Expected ip_hidden_states shape: (batch_size, query_seq_len, num_heads * head_dim)
            hidden_states = hidden_states + ipa_scale * ip_hidden_states * ip_mask

        return hidden_states

    def run_attention(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.attention_type == "torch-sdp":
            attn_call = self.run_attention_sdp
        elif self.attention_type == "sliced":
            attn_call = self.run_attention_sliced
        else:
            raise Exception(f"Unknown attention type: {self.attention_type}")

        return attn_call(
            attn=attn,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

    def run_attention_sdp(
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
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        return hidden_states

    def run_attention_sliced(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # slice_size
        if self.slice_size == "max":
            slice_size = 1
        elif self.slice_size == "auto":
            slice_size = max(1, attn.sliceable_head_dim // 2)
        else:
            slice_size = min(self.slice_size, attn.sliceable_head_dim)

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
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)
        return hidden_states
