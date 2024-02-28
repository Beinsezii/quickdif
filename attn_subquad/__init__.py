import torch
from torch import Tensor
from diffusers.models.attention import Attention
from typing import Optional
from .sub_quadratic_attention import efficient_dot_product_attention

class SubQuadraticCrossAttnProcessor:
    query_chunk_size: int
    kv_chunk_size: Optional[int]
    kv_chunk_size_min: Optional[int]
    chunk_threshold_bytes: Optional[int]
    def __init__(
        self,
        query_chunk_size = 1024,
        kv_chunk_size: Optional[int] = None,
        kv_chunk_size_min: Optional[int] = None,
        chunk_threshold_bytes: Optional[int] = None,
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
        self.chunk_threshold_bytes = chunk_threshold_bytes

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor]=None,
        attention_mask: Optional[Tensor]=None,
    ):
        encoder_hidden_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        assert attention_mask is None, "attention-mask not currently implemented for SubQuadraticCrossAttnProcessor."
        # I don't know what test case can be used to determine whether softmax is computed at sufficient bit-width,
        # but sub-quadratic attention has a pretty bespoke softmax (defers computation of the denominator) so this needs some thought.
        assert not attn.upcast_softmax or torch.finfo(hidden_states.dtype).bits >= 32, "upcast_softmax was requested, but is not implemented"

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.unflatten(-1, (attn.heads, -1)).transpose(1,2).flatten(end_dim=1)
        key_t = key.transpose(1,2).unflatten(1, (attn.heads, -1)).flatten(end_dim=1)
        del key
        value = value.unflatten(-1, (attn.heads, -1)).transpose(1,2).flatten(end_dim=1)

        dtype = query.dtype
        # TODO: do we still need to do *everything* in float32, given how we delay the division?
        # TODO: do we need to support upcast_softmax too? SD 2.1 seems to work without it
        if attn.upcast_attention:
            query = query.float()
            key_t = key_t.float()

        bytes_per_token = torch.finfo(query.dtype).bits//8
        batch_x_heads, q_tokens, _ = query.shape
        _, _, k_tokens = key_t.shape
        qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

        query_chunk_size = self.query_chunk_size
        kv_chunk_size = self.kv_chunk_size

        if self.chunk_threshold_bytes is not None and qk_matmul_size_bytes <= self.chunk_threshold_bytes:
            # the big matmul fits into our memory limit; do everything in 1 chunk,
            # i.e. send it down the unchunked fast-path
            query_chunk_size = q_tokens
            kv_chunk_size = k_tokens

        hidden_states = efficient_dot_product_attention(
            query,
            key_t,
            value,
            query_chunk_size=query_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min=self.kv_chunk_size_min,
            use_checkpoint=attn.training,
        )

        hidden_states = hidden_states.to(dtype)

        hidden_states = hidden_states.unflatten(0, (-1, attn.heads)).transpose(1,2).flatten(start_dim=2)

        out_proj, dropout = attn.to_out
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        return hidden_states
