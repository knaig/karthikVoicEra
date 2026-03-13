"""Delay-pattern helpers for staggered multi-codebook generation (Parler-TTS)."""

from typing import Tuple

import torch


def apply_delay_pattern_mask(
    input_ids: torch.Tensor,
    decoder_pad_token_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply the delay-pattern mask to decoder input ids.
    Positions where mask == -1 keep input_ids; others are replaced by the mask value (BOS/pad).
    """
    seq_len = input_ids.shape[-1]
    decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
    return torch.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)


def build_delay_pattern_mask(
    input_ids: torch.LongTensor,
    bos_token_id: int,
    pad_token_id: int,
    max_length: int,
    num_codebooks: int,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Build the staggered codebook pattern: each codebook is offset by one position.
    Returns (trimmed input_ids, full pattern mask) for use in generation.
    """
    input_ids = input_ids.reshape(-1, num_codebooks, input_ids.shape[-1])
    bsz, num_codebooks, seq_len = input_ids.shape

    input_ids_shifted = torch.ones(
        (bsz, num_codebooks, max_length), dtype=torch.long, device=input_ids.device
    ) * -1

    if max_length < 2 * num_codebooks - 1:
        return input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1)

    for codebook in range(num_codebooks):
        input_ids_shifted[:, codebook, codebook : seq_len + codebook] = input_ids[:, codebook]

    eos_delay_pattern = torch.triu(
        torch.ones((num_codebooks, max_length), dtype=torch.bool),
        diagonal=max_length - num_codebooks + 1,
    )
    bos_delay_pattern = torch.tril(torch.ones((num_codebooks, max_length), dtype=torch.bool))

    bos_mask = ~bos_delay_pattern.to(input_ids.device)
    eos_mask = ~eos_delay_pattern.to(input_ids.device)
    mask = ~(bos_delay_pattern + eos_delay_pattern).to(input_ids.device)
    input_ids = mask * input_ids_shifted + ~bos_mask * bos_token_id + ~eos_mask * pad_token_id

    first_codebook_ids = input_ids[:, 0, :]
    start_ids = (first_codebook_ids == -1).nonzero()[:, 1]
    first_start_id = min(start_ids) if len(start_ids) > 0 else seq_len

    pattern_mask = input_ids.reshape(bsz * num_codebooks, -1)
    input_ids = input_ids[..., :first_start_id].reshape(bsz * num_codebooks, -1)
    return input_ids, pattern_mask
