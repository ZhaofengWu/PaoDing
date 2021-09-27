import argparse
import logging

from allennlp.nn.util import masked_max, masked_mean
import torch

logger = logging.getLogger(__name__)

POOLING_MODES = {"avg", "max", "last"}


class Pooler(torch.nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        assert hparams.pooling_mode in POOLING_MODES

        super().__init__()
        self.hparams = hparams

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor = None):
        """
        Input:
            tensor: (bsz, seq_len, emb_dim)
            mask: (bsz, seq_len)

        output:
            pooled_tensor: (bsz, emb_dim)
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
        elif self.hparams.pooling_mode == "last":
            if mask is None:
                return tensor[:, -1, :]
            if mask is not None:
                last_pos = mask.sum(-1) - 1  # (bsz,)
                return tensor[torch.arange(tensor.shape[0]), last_pos]
        else:
            raise KeyError("Unsupported pooling mode")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--pooling_mode",
            default=None,
            type=str,
            required=True,
            choices=POOLING_MODES,
        )

        return parser
