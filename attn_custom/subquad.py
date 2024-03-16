from typing import Optional

import torch

from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate

from .sub_quadratic_attention import efficient_dot_product_attention


class SubQuadraticCrossAttnProcessor:
    query_chunk_size: int
    kv_chunk_size: Optional[int]
    kv_chunk_size_min: Optional[int]

    def __init__(
        self,
        query_chunk_size=1024,
        kv_chunk_size: Optional[int] = None,
        kv_chunk_size_min: Optional[int] = None,
    ):
        r"""
        Args:
            query_chunk_size (`int`, *optional*, defaults to `1024`)
            kv_chunk_size (`int`, *optional*, defaults to `None`): if None, sqrt(key_tokens) is used.
            kv_chunk_size_min (`int`, *optional*, defaults to `None`): only considered when `kv_chunk_size is None`. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
            chunk_threshold_bytes (`int`, *optional*, defaults to `None`): if defined: only bother chunking if the self-attn matmul would allocate more bytes than this. whenever we can fit traditional attention into memory: we should prefer to do so, as the unchunked algorithm is faster.
        """
        self.query_chunk_size = query_chunk_size
        self.kv_chunk_size = kv_chunk_size
        self.kv_chunk_size_min = kv_chunk_size_min

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).transpose(1, 2).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = efficient_dot_product_attention(
            query=query,
            key_t=key,
            value=value,
            query_chunk_size=self.query_chunk_size,
            kv_chunk_size=self.kv_chunk_size,
            kv_chunk_size_min=self.kv_chunk_size_min,
            use_checkpoint=False,
            upcast_attention=attn.upcast_attention,
            mask=attention_mask,
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

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
