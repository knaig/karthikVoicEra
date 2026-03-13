"""Docstring constants for Parler-TTS model docs (HF-style)."""

CONFIG_FOR_DOC = "ParlerTTSConfig"

MUSICGEN_START_DOCSTRING = r"""
    The ParlerTTS model was proposed in [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284).
    It is an encoder-decoder transformer for conditional music/speech generation.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for generic methods
    (downloading, saving, resizing embeddings, etc.).

    Parameters:
        config ([`ParlerTTSConfig`]): Model configuration. Only configuration is loaded; use
            [`~PreTrainedModel.from_pretrained`] to load weights.
"""

MUSICGEN_DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size * num_codebooks, sequence_length)`):
            Indices of input sequence tokens (audio codes).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask for padding token indices (1 = not masked, 0 = masked).
        encoder_hidden_states (`torch.FloatTensor`, *optional*):
            Encoder last hidden state for cross-attention.
        encoder_attention_mask (`torch.LongTensor`, *optional*):
            Mask for encoder padding.
        prompt_hidden_states / prompt_attention_mask: Prompt conditioning.
        head_mask / cross_attn_head_mask: Head masking.
        past_key_values: Cached key/value for decoding.
        inputs_embeds: Optional embedded inputs.
        output_attentions / output_hidden_states / return_dict: Output options.
"""
