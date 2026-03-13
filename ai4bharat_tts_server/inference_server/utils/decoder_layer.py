"""Single decoder layer: self-attn, cross-attn, and FFN."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import EncoderDecoderCache

from ..configuration_parler_tts import ParlerTTSDecoderConfig
from .attention import get_attention_class


class ParlerTTSDecoderLayer(nn.Module):
    """One decoder block: self-attention, cross-attention to encoder, and feed-forward."""

    def __init__(self, config: ParlerTTSDecoderConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        AttnClass = get_attention_class(config._attn_implementation)
        self.self_attn = AttnClass(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            bias=False,
            rope_embeddings=config.rope_embeddings,
            layer_idx=layer_idx,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        cross_impl = config._attn_implementation
        if config.cross_attention_implementation_strategy == "always_eager":
            cross_impl = "eager"
        elif config.cross_attention_implementation_strategy == "always_sdpa":
            cross_impl = "sdpa"
        self.encoder_attn = get_attention_class(cross_impl)(
            self.embed_dim,
            config.num_attention_heads,
            num_key_value_heads=config.num_cross_attention_key_value_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=False,
            rope_embeddings=config.rope_embeddings,
            layer_idx=layer_idx,
            config=config,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=False)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.LongTensor] = None,
        sin: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            cos=cos,
            sin=sin,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights, cross_attn_present = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                cos=cos,
                sin=sin,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            present_key_value = (present_key_value, cross_attn_present)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
