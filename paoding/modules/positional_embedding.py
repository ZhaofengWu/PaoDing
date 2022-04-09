import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def PositionalEmbedding(num_embeddings: int, embedding_dim: int, learned: bool = False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)  # from fairseq
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, init_size=num_embeddings + 1)
    return m


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        # HF BERT does this; I think this is to make device moving automatic
        self.register_buffer("position_ids", torch.arange(num_embeddings))

    def forward(self, input: Tensor, position_ids: Tensor = None):
        """
        input: (bsz, seq_len)
        output: (seq_len, emb_dim)
        """
        return F.embedding(
            self.position_ids[: input.shape[1]] if position_ids is None else position_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Adapted from the fairseq implementation.
    """

    def __init__(self, embedding_dim, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        # (num_emb, emb_dim)
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        return emb

    def forward(self, input: Tensor):
        """
        input: (bsz, seq_len)
        output: (seq_len, emb_dim)
        """
        max_pos = input.shape[1] + 1
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embedding_dim)
        self.weights = self.weights.to(self._float_tensor)

        positions = torch.arange(input.shape[1], device=self._float_tensor.device)
        return self.weights[positions].detach()
