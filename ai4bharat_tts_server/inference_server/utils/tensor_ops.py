"""Small tensor utilities for attention and labels."""

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for GQA/MQA.
    (batch, n_kv_heads, seq_len, head_dim) -> (batch, n_heads, seq_len, head_dim).
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def shift_tokens_right(
    input_ids: torch.Tensor,
    pad_token_id: int,
    decoder_start_token_id: int,
) -> torch.Tensor:
    """
    Shift input ids one position right and prepend decoder_start_token_id.
    Used for teacher-forcing when labels are provided.
    """
    shifted = input_ids.new_zeros(input_ids.shape)
    shifted[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("decoder_start_token_id must be set.")
    shifted[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("pad_token_id must be set.")
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted
