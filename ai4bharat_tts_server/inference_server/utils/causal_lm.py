"""Decoder model and causal LM with per-codebook LM heads and delay-pattern support."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from ..configuration_parler_tts import ParlerTTSDecoderConfig
from .base import ParlerTTSPreTrainedModel
from .decoder import ParlerTTSDecoder
from .delay_pattern import apply_delay_pattern_mask, build_delay_pattern_mask
from .docstrings import CONFIG_FOR_DOC, MUSICGEN_DECODER_INPUTS_DOCSTRING, MUSICGEN_START_DOCSTRING
from .outputs import ParlerTTSCausalLMOutputWithCrossAttentions


class ParlerTTSModel(ParlerTTSPreTrainedModel):
    """Bare decoder outputting hidden states (no LM head)."""

    def __init__(self, config: ParlerTTSDecoderConfig):
        super().__init__(config)
        self.decoder = ParlerTTSDecoder(config)
        self.config = config
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        prompt_hidden_states: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[object, Tuple]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        return self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            prompt_hidden_states=prompt_hidden_states,
            prompt_attention_mask=prompt_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


@add_start_docstrings(
    "Parler-TTS decoder with language modeling head on top.",
    MUSICGEN_START_DOCSTRING,
)
class ParlerTTSForCausalLM(ParlerTTSPreTrainedModel):
    """Decoder with per-codebook LM heads; used as the decoder in ParlerTTSForConditionalGeneration."""

    def __init__(self, config: ParlerTTSDecoderConfig):
        super().__init__(config)
        self.model = ParlerTTSModel(config)
        self.num_codebooks = config.num_codebooks
        self.vocab_size = config.vocab_size
        self.use_fused_lm_heads = config.use_fused_lm_heads
        if self.use_fused_lm_heads:
            self.lm_heads = nn.Linear(
                config.hidden_size, config.vocab_size * config.num_codebooks, bias=False
            )
        else:
            self.lm_heads = nn.ModuleList(
                [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(config.num_codebooks)]
            )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_heads

    def set_output_embeddings(self, new_embeddings):
        self.lm_heads = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def build_delay_pattern_mask(
        self,
        input_ids: torch.LongTensor,
        bos_token_id: int,
        pad_token_id: int,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Build staggered codebook delay mask for generation. Returns (trimmed_input_ids, pattern_mask)."""
        max_length = max_length or getattr(self.generation_config, "max_length", None)
        return build_delay_pattern_mask(
            input_ids, bos_token_id, pad_token_id, max_length, self.num_codebooks
        )

    @staticmethod
    def apply_delay_pattern_mask(input_ids: torch.Tensor, decoder_pad_token_mask: torch.Tensor) -> torch.Tensor:
        """Apply delay-pattern mask to decoder input ids (keep only positions where mask == -1)."""
        return apply_delay_pattern_mask(input_ids, decoder_pad_token_mask)

    @add_start_docstrings_to_model_forward(MUSICGEN_DECODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ParlerTTSCausalLMOutputWithCrossAttentions, config_class=CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        prompt_hidden_states: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        loss_reduction: str = "mean",
    ) -> Union[Tuple, ParlerTTSCausalLMOutputWithCrossAttentions]:
        r"""
        labels: optional LM labels (shifted inside). Shape (batch, seq_len, num_codebooks).
        Returns:
            ParlerTTSCausalLMOutputWithCrossAttentions or tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            prompt_hidden_states=prompt_hidden_states,
            prompt_attention_mask=prompt_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        if self.use_fused_lm_heads:
            lm_logits = self.lm_heads(hidden_states).view(
                hidden_states.shape[0], -1, self.num_codebooks, self.vocab_size
            ).transpose(1, 2)
        else:
            lm_logits = torch.stack([h(hidden_states) for h in self.lm_heads], dim=1)

        loss = None
        per_codebook_losses = None
        if labels is not None:
            codebook_weights = self.config.codebook_weights
            logits = lm_logits[:, :, -labels.shape[1] :]
            loss_fct = CrossEntropyLoss(reduction=loss_reduction)
            loss = torch.zeros([], device=self.device)
            per_codebook_losses = []
            labels = labels.masked_fill(labels == self.config.bos_token_id, -100)
            mask = (input_ids.transpose(1, 2) != self.config.eos_token_id) & (labels != -100)
            for codebook in range(self.config.num_codebooks):
                cb_logits = logits[:, codebook].contiguous().view(-1, logits.shape[-1])
                cb_mask = mask[..., codebook].contiguous().view(-1)
                cb_labels = labels[..., codebook].contiguous().view(-1)
                cb_loss = loss_fct(cb_logits[cb_mask], cb_labels[cb_mask])
                per_codebook_losses.append(cb_loss)
                if codebook_weights is not None:
                    cb_loss = cb_loss * codebook_weights[codebook]
                loss = loss + cb_loss
            loss = loss / (sum(codebook_weights) if codebook_weights is not None else self.config.num_codebooks)

        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])
        if not return_dict:
            out = (lm_logits,) + outputs[1:]
            return ((loss,) + out + (per_codebook_losses,)) if loss is not None else out
        return ParlerTTSCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            per_codebook_losses=per_codebook_losses,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        prompt_hidden_states=None,
        prompt_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=True,
        delay_pattern_mask=None,
        cache_position=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """Prepare decoder inputs for the next generation step (delay pattern, cache trim, position_ids)."""
        if delay_pattern_mask is None:
            input_ids, delay_pattern_mask = self.build_delay_pattern_mask(
                input_ids,
                bos_token_id=self.generation_config.bos_token_id,
                pad_token_id=self.generation_config.pad_token_id,
                max_length=self.generation_config.max_length,
            )
        input_ids = self.apply_delay_pattern_mask(input_ids, delay_pattern_mask)
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]
            prompt_hidden_states = None
        return {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "prompt_hidden_states": prompt_hidden_states,
            "prompt_attention_mask": prompt_attention_mask,
            "head_mask": head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "delay_pattern_mask": delay_pattern_mask,
            "cache_position": cache_position,
            "inputs_embeds": inputs_embeds,
        }
