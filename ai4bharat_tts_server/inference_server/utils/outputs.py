"""Model output types for Parler-TTS (seq2seq and causal LM)."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class ParlerTTSSeq2SeqLMOutput(ModelOutput):
    """
    Full encoder-decoder forward output (description + prompt → logits).

    Args:
        loss: LM loss when labels are provided.
        logits: Prediction scores from the LM head.
        past_key_values: Cached KV for decoding.
        decoder_hidden_states: Decoder layer hidden states.
        decoder_attentions: Decoder self-attention weights.
        cross_attentions: Cross-attention to encoder.
        encoder_last_hidden_state: Last encoder hidden state.
        encoder_hidden_states: Encoder layer hidden states.
        encoder_attentions: Encoder attention weights.
        per_codebook_losses: Per-codebook loss terms.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    per_codebook_losses: Optional[List[torch.FloatTensor]] = None


@dataclass
class ParlerTTSCausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Decoder-only forward output (causal LM with cross-attention).

    Args:
        loss: LM loss when labels are provided.
        logits: Prediction scores from the LM head.
        past_key_values: Cached KV for decoding.
        hidden_states: Decoder layer hidden states.
        attentions: Self-attention weights.
        cross_attentions: Cross-attention to encoder.
        per_codebook_losses: Per-codebook loss terms.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    per_codebook_losses: Optional[List[torch.FloatTensor]] = None
