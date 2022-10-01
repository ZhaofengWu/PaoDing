import argparse
import logging

import torch
from torch import nn

from paoding.argument_parser import ArgumentParser
from paoding.nn_utils import masked_max, masked_mean, masked_softmax, weighted_sum

logger = logging.getLogger(__name__)

POOLING_MODES = {"avg", "max", "first", "last", "attn", "attn_k", "attn_v", "attn_kv"}


class Pooler(nn.Module):
    def __init__(self, hparams: argparse.Namespace, hidden_dim: int = None):
        assert hparams.pooling_mode in POOLING_MODES

        super().__init__()
        self.hparams = hparams
        if self.hparams.pooling_mode in {"attn", "attn_k", "attn_v", "attn_kv"}:
            assert hidden_dim is not None
            self.q = nn.Linear(hidden_dim, 1, bias=False)
            if self.hparams.pooling_mode in {"attn_k", "attn_kv"}:
                self.K = nn.Linear(hidden_dim, hidden_dim)
            if self.hparams.pooling_mode in {"attn_v", "attn_kv"}:
                self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None):
        """
        Input:
            tensor: (bsz, seq_len, hidden_dim)
            mask: (bsz, seq_len)

        output:
            pooled_tensor: (bsz, hidden_dim)
        """
        if self.hparams.pooling_mode == "avg":
            if mask is None:
                return tensor.mean(1)
            else:
                return masked_mean(tensor, mask.unsqueeze(-1), 1)
        elif self.hparams.pooling_mode == "max":
            if mask is None:
                return tensor.max(1)
            else:
                return masked_max(tensor, mask.unsqueeze(-1), 1)
        elif self.hparams.pooling_mode == "first":
            assert mask[:, 0].all()
            return tensor[:, 0, :]
        elif self.hparams.pooling_mode == "last":
            if mask is None:
                return tensor[:, -1, :]
            if mask is not None:
                last_pos = mask.sum(-1) - 1  # (bsz,)
                return tensor[torch.arange(tensor.shape[0]), last_pos]
        elif self.hparams.pooling_mode in {"attn", "attn_k", "attn_v", "attn_kv"}:
            k = v = tensor  # (bsz, seq_len, hidden_dim)
            if self.hparams.pooling_mode in {"attn_k", "attn_kv"}:
                k = self.K(k)
            if self.hparams.pooling_mode in {"attn_v", "attn_kv"}:
                v = self.V(v)
            scores = self.q(k).squeeze(-1)  # (bsz, seq_len)
            # masked_softmax handles None mask automatically, and makes sure that paddings are 0
            weights = masked_softmax(scores, mask, dim=-1)  # (bsz, seq_len)
            return weighted_sum(v, weights)
        else:
            raise KeyError("Unsupported pooling mode")

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--pooling_mode",
            default=None,
            type=str,
            required=True,
            choices=POOLING_MODES,
        )

        return parser
