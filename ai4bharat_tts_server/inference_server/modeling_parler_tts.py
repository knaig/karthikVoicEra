# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""
Parler-TTS encoder-decoder model: text encoder + audio encoder + transformer decoder.

This module is the main entry point. Decoder stack, outputs, delay-pattern and
generation helpers live in inference_server.utils (see utils/__init__.py).
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForTextEncoding
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from importlib.metadata import version
from packaging.version import Version

from .configuration_parler_tts import ParlerTTSConfig, ParlerTTSDecoderConfig
from .dac_wrapper import DACConfig, DACModel
from .logits_processors import ParlerTTSLogitsProcessor
from .utils import (
    ParlerTTSForCausalLM,
    ParlerTTSSeq2SeqLMOutput,
    shift_tokens_right,
)
from .utils.generation import (
    get_cache,
    get_decoder_start_token_id,
    get_initial_cache_position,
    maybe_initialize_input_ids_for_generation,
    prepare_audio_encoder_kwargs_for_generation,
    prepare_decoder_input_ids_for_generation,
    prepare_prompt_kwargs_for_generation,
    prepare_text_encoder_kwargs_for_generation,
)

# DAC registration (transformers version–dependent)
_is_dac_integrated = Version(version("transformers")) > Version("4.44.2dev")
if not _is_dac_integrated:
    AutoConfig.register("dac", DACConfig)
else:
    AutoConfig.register("dac_on_the_hub", DACConfig)
AutoModel.register(DACConfig, DACModel)

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "ParlerTTSConfig"
from transformers.cache_utils import SlidingWindowCache, StaticCache
NEED_SETUP_CACHE_CLASSES_MAPPING = {"static": StaticCache, "sliding_window": SlidingWindowCache}

INPUTS_DOCSTRING = r"""
    Args:
        input_ids: Text description token ids. attention_mask: Optional mask.
        input_values / padding_mask: Audio for conditioning.
        decoder_input_ids: Decoder audio codes (optional).
        prompt_input_ids / prompt_attention_mask: Prompt text to speak.
        encoder_outputs: Precomputed encoder output (optional).
        past_key_values: KV cache for decoding.
        labels: For training (shifted inside).
        use_cache, output_attentions, output_hidden_states, return_dict, cache_position, loss_reduction.
"""


@add_start_docstrings(
    "Parler-TTS: text encoder + audio encoder + decoder for conditional TTS.",
    r"""
    This model inherits from [`PreTrainedModel`]. See superclass for generic methods.
    Parameters:
        config ([`ParlerTTSConfig`]): Model configuration.
    """,
)
class ParlerTTSForConditionalGeneration(PreTrainedModel):
    """Encoder-decoder TTS: text and/or audio conditioning → decoder logits."""

    config_class = ParlerTTSConfig
    base_model_prefix = "encoder_decoder"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(
        self,
        config: Optional[ParlerTTSConfig] = None,
        text_encoder: Optional[PreTrainedModel] = None,
        audio_encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[ParlerTTSForCausalLM] = None,
    ):
        if config is None and (text_encoder is None or audio_encoder is None or decoder is None):
            raise ValueError(
                "Provide config or all of text_encoder, audio_encoder, decoder."
            )
        if config is None:
            config = ParlerTTSConfig.from_sub_models_config(
                text_encoder.config, audio_encoder.config, decoder.config
            )
        elif not isinstance(config, self.config_class):
            raise ValueError(f"Config must be of type {self.config_class}")
        if getattr(config.decoder, "cross_attention_hidden_size", None) is not None:
            if config.decoder.cross_attention_hidden_size != config.text_encoder.hidden_size:
                raise ValueError(
                    "decoder.cross_attention_hidden_size must equal text_encoder.hidden_size when set."
                )
        super().__init__(config)
        if text_encoder is None:
            text_encoder = AutoModelForTextEncoding.from_config(config.text_encoder)
        if audio_encoder is None:
            audio_encoder = AutoModel.from_config(config.audio_encoder)
        if decoder is None:
            decoder = ParlerTTSForCausalLM(config.decoder)
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder
        for name, sub in [("text_encoder", text_encoder), ("audio_encoder", audio_encoder), ("decoder", decoder)]:
            if sub.config.to_dict() != getattr(self.config, name).to_dict():
                logger.warning(f"Config of {name} overwritten by shared config.")
        self.config.text_encoder._attn_implementation = self.text_encoder.config._attn_implementation
        self.config.audio_encoder._attn_implementation = self.audio_encoder.config._attn_implementation
        self.config.decoder._attn_implementation = self.decoder.config._attn_implementation
        self.text_encoder.config = self.config.text_encoder
        self.audio_encoder.config = self.config.audio_encoder
        self.decoder.config = self.config.decoder
        if (
            self.text_encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(
                self.text_encoder.config.hidden_size, self.decoder.config.hidden_size
            )
        self.embed_prompts = nn.Embedding(config.vocab_size, self.decoder.config.hidden_size)
        if self.text_encoder.get_output_embeddings() is not None:
            raise ValueError("Text encoder must not have an LM head.")
        if "encoder_hidden_states" not in set(inspect.signature(self.decoder.forward).parameters):
            raise ValueError("Decoder must accept encoder_hidden_states.")
        self.use_audio_scales = "audio_scales" in set(
            inspect.signature(self.audio_encoder.decode).parameters
        )
        self.use_4dim_audio_codes = False
        at = audio_encoder.config.model_type
        if at in {"encodec", "dac_on_the_hub"} or (at == "dac" and not _is_dac_integrated):
            self.use_4dim_audio_codes = True
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.decoder.config.initializer_factor
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def tie_weights(self) -> None:
        if self.config.tie_encoder_decoder:
            dec_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.text_encoder,
                self.decoder._modules[dec_prefix],
                dec_prefix,
            )

    def get_audio_encoder(self):
        return self.audio_encoder

    def get_text_encoder(self):
        return self.text_encoder

    def get_encoder(self):
        return self.get_text_encoder()

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.text_encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        if kwargs.get("_fast_init", False):
            logger.warning("Fast init not supported for ParlerTTSForConditionalGeneration; using slow init.")
        kwargs["_fast_init"] = False
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @add_start_docstrings_to_model_forward(INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ParlerTTSSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        input_values: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.BoolTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        prompt_input_ids: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        prompt_hidden_states: Optional[torch.FloatTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        loss_reduction: str = "mean",
        **kwargs,
    ) -> Union[Tuple, ParlerTTSSeq2SeqLMOutput]:
        r"""
        Returns:
            ParlerTTSSeq2SeqLMOutput or tuple of (loss, logits, ...).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs_text = {k[len("text_encoder_"):]: v for k, v in kwargs.items() if k.startswith("text_encoder_")}
        kwargs_audio = {k[len("audio_encoder_"):]: v for k, v in kwargs.items() if k.startswith("audio_encoder_")}
        kwargs_decoder = {k[len("decoder_"):]: v for k, v in kwargs.items() if k.startswith("decoder_")}
        if prompt_hidden_states is None and prompt_input_ids is not None:
            prompt_hidden_states = self.embed_prompts(prompt_input_ids)
        if encoder_outputs is None:
            encoder_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_text,
            )
            encoder_hidden_states = encoder_outputs[0]
            if (
                self.text_encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
            ):
                encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
            if attention_mask is not None:
                encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            ).transpose(1, 2)
        elif decoder_input_ids is None and decoder_inputs_embeds is None:
            audio_out = self.audio_encoder(
                input_values=input_values, padding_mask=padding_mask, **kwargs_audio
            )
            audio_codes = audio_out.audio_codes
            frames, bsz, codebooks, seq_len = audio_codes.shape
            if frames != 1:
                raise ValueError("Expected 1 frame; disable chunking (chunk_length=None).")
            if self.config.decoder.audio_channels == 2 and audio_codes.shape[2] == self.decoder.num_codebooks // 2:
                audio_codes = audio_codes.repeat_interleave(2, dim=2)
            decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            prompt_hidden_states=prompt_hidden_states,
            prompt_attention_mask=prompt_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            cache_position=cache_position,
            loss_reduction=loss_reduction,
            **kwargs_decoder,
        )
        if not return_dict:
            return decoder_outputs + (encoder_hidden_states,)
        return ParlerTTSSeq2SeqLMOutput(
            loss=decoder_outputs.loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=getattr(encoder_outputs, "hidden_states", None),
            encoder_attentions=getattr(encoder_outputs, "attentions", None),
            per_codebook_losses=decoder_outputs.per_codebook_losses,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        prompt_hidden_states=None,
        prompt_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        decoder_delay_pattern_mask=None,
        cache_position=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if decoder_delay_pattern_mask is None:
            decoder_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
                decoder_input_ids,
                bos_token_id=self.generation_config.bos_token_id,
                pad_token_id=self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )
        decoder_input_ids = self.decoder.apply_delay_pattern_mask(
            decoder_input_ids, decoder_delay_pattern_mask
        )
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                past_length = (
                    cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                )
                if past_key_values.get_seq_length() > 0:
                    prompt_hidden_states = None
            else:
                past_length = past_key_values[0][0].shape[2]
                prompt_hidden_states = None
            remove = past_length if decoder_input_ids.shape[1] > past_length else decoder_input_ids.shape[1] - 1
            decoder_input_ids = decoder_input_ids[:, remove:]
        if cache_position is None:
            cache_position = torch.arange(
                past_length,
                past_length + decoder_input_ids.shape[1],
                device=decoder_input_ids.device,
            )
        elif use_cache:
            cur_len = decoder_input_ids.shape[1]
            if prompt_hidden_states is not None:
                cur_len += prompt_hidden_states.shape[1]
            cache_position = cache_position[-cur_len:]
        if decoder_attention_mask is None and prompt_attention_mask is not None:
            input_ = decoder_input_ids.reshape(-1, self.decoder.num_codebooks, decoder_input_ids.shape[-1])
            bsz, _, seq_len = input_.shape
            input_shape = (bsz, seq_len)
            past_len = cache_position[0] if cache_position is not None else (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            logger.warning_once(
                "prompt_attention_mask set but decoder_attention_mask not; creating full mask."
            )
            if past_key_values is None or (
                isinstance(past_key_values, EncoderDecoderCache) and past_key_values.get_seq_length() == 0
            ):
                decoder_attention_mask = torch.ones(
                    input_shape, device=self.device, dtype=decoder_input_ids.dtype
                )
            else:
                gen_len = past_len - prompt_attention_mask.shape[1] + 1
                decoder_attention_mask = torch.ones(
                    (input_shape[0], gen_len), device=self.device, dtype=prompt_attention_mask.dtype
                )
        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids.contiguous(),
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "prompt_hidden_states": prompt_hidden_states,
            "prompt_attention_mask": prompt_attention_mask,
            "use_cache": use_cache,
            "cache_position": cache_position,
            "inputs_embeds": inputs_embeds,
        }

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, Any],
        decoder_start_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        return prepare_decoder_input_ids_for_generation(
            self, batch_size, model_input_name, model_kwargs,
            decoder_start_token_id, bos_token_id, device,
        )

    def _prepare_text_encoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs: Dict,
        model_input_name: Optional[str],
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        return prepare_text_encoder_kwargs_for_generation(
            self, inputs_tensor, model_kwargs, model_input_name, generation_config,
        )

    def _prepare_prompt_kwargs_for_generation(self, prompt_input_ids, model_kwargs):
        return prepare_prompt_kwargs_for_generation(self, prompt_input_ids, model_kwargs)

    def _prepare_audio_encoder_kwargs_for_generation(
        self, input_values, model_kwargs, model_input_name: Optional[str] = None
    ):
        return prepare_audio_encoder_kwargs_for_generation(
            self, input_values, model_kwargs, model_input_name,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError(
            "Resize embeddings on encoder or decoder directly, not on this model."
        )

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        return maybe_initialize_input_ids_for_generation(
            self, inputs, bos_token_id, model_kwargs or {},
        )

    def _get_decoder_start_token_id(
        self,
        decoder_start_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
    ) -> int:
        return get_decoder_start_token_id(self, decoder_start_token_id, bos_token_id)

    def _get_cache(
        self,
        cache_implementation: str,
        max_batch_size: int,
        max_cache_len: int,
        model_kwargs: Dict,
    ) -> Cache:
        return get_cache(
            self, NEED_SETUP_CACHE_CLASSES_MAPPING,
            cache_implementation, max_batch_size, max_cache_len, model_kwargs,
        )

    def _get_initial_cache_position(self, input_ids, model_kwargs):
        return get_initial_cache_position(self, input_ids, model_kwargs)
