"""Multi-head attention (eager and SDPA) with GQA/MQA support for the decoder."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import EncoderDecoderCache
from transformers.utils import logging

from ..configuration_parler_tts import ParlerTTSDecoderConfig
from .tensor_ops import repeat_kv

logger = logging.get_logger(__name__)


def get_attention_class(attn_implementation: str):
    """Return the attention module class for the given implementation (eager/sdpa/flash_attention_2)."""
    if attn_implementation == "flash_attention_2":
        return ParlerTTSSdpaAttention
    return {"eager": ParlerTTSAttention, "sdpa": ParlerTTSSdpaAttention}[attn_implementation]


class ParlerTTSAttention(nn.Module):
    """Multi-headed attention with optional GQA/MQA (key/value head grouping)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        rope_embeddings: bool = False,
        layer_idx: Optional[int] = None,
        config: Optional[ParlerTTSDecoderConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.config = config
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got embed_dim={self.embed_dim}, num_heads={num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        if layer_idx is None and is_decoder:
            logger.warning_once(
                f"Instantiating decoder {self.__class__.__name__} without layer_idx is not recommended when using cache."
            )
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape_query(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _shape_key_value(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.LongTensor] = None,
        sin: Optional[torch.LongTensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[EncoderDecoderCache]]:
        is_cross_attention = key_value_states is not None
        bsz, tgt_len = hidden_states.shape[:2]

        query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self._shape_query(query_states, tgt_len, bsz)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self._shape_key_value(self.k_proj(current_states), -1, bsz)
            value_states = self._shape_key_value(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be size {(self.num_heads,)}, got {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_probs, value_states)
        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"attn_output should be size {(bsz, self.num_heads, tgt_len, self.head_dim)}, got {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class ParlerTTSSdpaAttention(ParlerTTSAttention):
    """SDPA-backed attention; falls back to eager when output_attentions or head_mask is used."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.LongTensor] = None,
        sin: Optional[torch.LongTensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[EncoderDecoderCache]]:
        if output_attentions or layer_head_mask is not None:
            logger.warning_once(
                "ParlerTTSModel is using ParlerTTSSdpaAttention but output_attentions or layer_head_mask is set; "
                "falling back to eager attention."
            )
            return super().forward(
                hidden_states,
                key_value_states=key_value_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
        is_cross_attention = key_value_states is not None
        bsz, tgt_len = hidden_states.shape[:2]
        query_states = self.q_proj(hidden_states)
        query_states = self._shape_query(query_states, tgt_len, bsz)

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                past_key_value.is_updated[self.layer_idx] = True
                past_key_value = past_key_value.cross_attention_cache
            else:
                past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if key_value_states is not None else hidden_states
        if is_cross_attention and past_key_value and is_updated:
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self._shape_key_value(self.k_proj(current_states), -1, bsz)
            value_states = self._shape_key_value(self.v_proj(current_states), -1, bsz)
            if past_key_value is not None:
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]] if attention_mask is not None else None
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        is_causal = bool(self.is_causal and causal_mask is None and tgt_len > 1)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"attn_output should be size {(bsz, self.num_heads, tgt_len, self.head_dim)}, got {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value
