"""Sinusoidal positional embeddings for the decoder."""

import math
import torch
import torch.nn as nn


class ParlerTTSSinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings of arbitrary length (no learned weights)."""

    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.offset = 0
        self._make_weights(num_positions, embedding_dim)

    def _make_weights(self, num_embeddings: int, embedding_dim: int) -> None:
        emb = self._get_embedding(num_embeddings, embedding_dim)
        if hasattr(self, "weights"):
            emb = emb.to(dtype=self.weights.dtype, device=self.weights.device)
        self.weights = nn.Parameter(emb)
        self.weights.requires_grad = False
        self.weights.detach_()

    @staticmethod
    def _get_embedding(num_embeddings: int, embedding_dim: int) -> torch.Tensor:
        half_dim = embedding_dim // 2
        inv_freq = math.log(10000) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -inv_freq)
        pos = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1)
        emb = pos * inv_freq.unsqueeze(0)
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0) -> torch.Tensor:
        bsz, seq_len, _ = input_ids.size()
        position_ids = torch.arange(seq_len, device=input_ids.device) + past_key_values_length
        if seq_len > self.weights.size(0):
            self._make_weights(seq_len + self.offset, self.embedding_dim)
        return self.weights.index_select(0, position_ids.view(-1)).detach()

    @torch.no_grad()
    def from_position_ids(self, position_ids: torch.Tensor) -> torch.Tensor:
        bs, seq_len = position_ids.shape
        flat = self.weights.index_select(0, position_ids.view(-1))
        return flat.reshape(bs, seq_len, -1)
