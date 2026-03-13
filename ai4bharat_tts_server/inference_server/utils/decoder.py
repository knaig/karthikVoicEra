"""Transformer decoder stack: layers, causal mask, and forward."""

import math
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    StaticCache,
)
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import add_start_docstrings_to_model_forward, logging

from ..configuration_parler_tts import ParlerTTSDecoderConfig
from .base import ParlerTTSPreTrainedModel
from .decoder_layer import ParlerTTSDecoderLayer
from .docstrings import MUSICGEN_DECODER_INPUTS_DOCSTRING
from .positional_embedding import ParlerTTSSinusoidalPositionalEmbedding

logger = logging.get_logger(__name__)


class ParlerTTSDecoder(ParlerTTSPreTrainedModel):
    """Transformer decoder: embeddings, sinusoidal positions, N decoder layers, layer norm."""

    def __init__(self, config: ParlerTTSDecoderConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.max_target_positions = config.max_position_embeddings
        self.d_model = config.hidden_size
        self.num_codebooks = config.num_codebooks
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        embed_dim = config.vocab_size + 1  # +1 for pad
        self.embed_tokens = nn.ModuleList(
            [nn.Embedding(embed_dim, config.hidden_size) for _ in range(config.num_codebooks)]
        )
        self.embed_positions = ParlerTTSSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [ParlerTTSDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.attn_implementation = config._attn_implementation
        enc_attn = config._attn_implementation
        if config.cross_attention_implementation_strategy is not None:
            enc_attn = "sdpa" if config.cross_attention_implementation_strategy == "always_sdpa" else "eager"
        self.encoder_attn_implementation = enc_attn
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _build_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        output_attentions: bool,
    ) -> Optional[torch.Tensor]:
        """Build 4D causal (+ optional padding) mask for self-attention."""
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static = isinstance(past_key_values, StaticCache)
        if self.config._attn_implementation == "sdpa" and not using_static and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen,
                is_training=self.training,
            ):
                return None
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        seq_len = input_tensor.shape[1]
        target_len = (
            past_key_values.get_max_length()
            if using_static
            else (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen + seq_len + 1
            )
        )
        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed inverted with max==0.")
            return attention_mask
        causal = torch.full((seq_len, target_len), min_dtype, dtype=dtype, device=device)
        if seq_len != 1:
            causal = torch.triu(causal, diagonal=1)
        causal *= torch.arange(target_len, device=device) > cache_position.reshape(-1, 1)
        causal = causal[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal = causal.clone()
            mask_len = attention_mask.shape[-1]
            pad_mask = (causal[:, :, :, :mask_len] + attention_mask[:, None, None, :]) == 0
            causal[:, :, :, :mask_len] = causal[:, :, :, :mask_len].masked_fill(pad_mask, min_dtype)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal = AttentionMaskConverter._unmask_unattended(causal, min_dtype)
        return causal

    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        prompt_hidden_states: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds.")
        if input_ids is not None:
            input = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
            bsz, num_codebooks, seq_len = input.shape
            input_shape = (bsz, seq_len)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1:]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = sum(
                self.embed_tokens[c](input[:, c]) for c in range(self.num_codebooks)
            )

        prepended_len = 0
        if prompt_hidden_states is not None:
            prepended_len = prompt_hidden_states.shape[-2]
            inputs_embeds = torch.cat([prompt_hidden_states, inputs_embeds], dim=1)

        return_legacy = False
        return_self_only = False
        if use_cache or past_key_values is not None:
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_only = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy = True
                logger.warning_once(
                    "Tuple past_key_values is deprecated; use EncoderDecoderCache."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        past_len = cache_position[0] if cache_position is not None else (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_len,
                past_len + input_shape[1] + prepended_len,
                device=inputs_embeds.device,
            )

        if prompt_attention_mask is not None and attention_mask is not None:
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        elif prompt_attention_mask is not None:
            logger.warning_once("prompt_attention_mask set but attention_mask not; creating full mask.")
            if past_len == 0:
                attention_mask = torch.cat(
                    [
                        prompt_attention_mask,
                        torch.ones(input_shape, device=self.device, dtype=prompt_attention_mask.dtype),
                    ],
                    dim=1,
                )
            else:
                gen_len = past_len - prompt_attention_mask.shape[1] + 1
                attention_mask = torch.cat(
                    [
                        prompt_attention_mask,
                        torch.ones(
                            (input_shape[0], gen_len),
                            device=self.device,
                            dtype=prompt_attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )

        input_shape = inputs_embeds.size()[:-1]
        if position_ids is None:
            positions = self.embed_positions(inputs_embeds, past_len)
        else:
            positions = self.embed_positions.from_position_ids(position_ids)
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._build_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values.self_attention_cache if past_key_values is not None else None,
            output_attentions,
        )

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # Only convert from 2D to 4D; if already 4D (e.g. from ragged runner), use as-is
            if encoder_attention_mask.ndim != 4:
                if (
                    self.encoder_attn_implementation == "sdpa"
                    and cross_attn_head_mask is None
                    and not output_attentions
                ):
                    encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                        encoder_attention_mask,
                        inputs_embeds.dtype,
                        tgt_len=input_shape[-1],
                    )
                else:
                    encoder_attention_mask = _prepare_4d_attention_mask(
                        encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                    )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("use_cache=True incompatible with gradient checkpointing; setting use_cache=False.")
            use_cache = False

        all_hidden = () if output_hidden_states else None
        all_self_attn = () if output_attentions else None
        all_cross_attn = () if (output_attentions and encoder_hidden_states is not None) else None
        for m, mname in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if m is not None and m.size()[0] != len(self.layers):
                raise ValueError(f"{mname} must be specified for {len(self.layers)} layers, got {m.size()[0]}.")

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden += (hidden_states,)
            if self.training and random.uniform(0, 1) < self.layerdrop:
                continue
            if self.gradient_checkpointing and self.training:
                layer_out = self._gradient_checkpointing_func(
                    layer.forward,
                    hidden_states,
                    causal_mask,
                    None,
                    None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_out = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    cos=None,
                    sin=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=head_mask[idx] if head_mask is not None else None,
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_values if use_cache else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            hidden_states = layer_out[0]
            if output_attentions:
                all_self_attn += (layer_out[1],)
                if encoder_hidden_states is not None:
                    all_cross_attn += (layer_out[2],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden += (hidden_states,)

        next_cache = past_key_values if use_cache else None
        if return_self_only:
            next_cache = past_key_values.self_attention_cache
        if return_legacy:
            next_cache = past_key_values.to_legacy_cache()
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden, all_self_attn, all_cross_attn] if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden,
            attentions=all_self_attn,
            cross_attentions=all_cross_attn,
        )
