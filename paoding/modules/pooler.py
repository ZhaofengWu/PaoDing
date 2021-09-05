import argparse
import logging

import torch

logger = logging.getLogger(__name__)

POOLING_MODES = {"avg", "max"}


class Pooler(torch.nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        assert hparams.pooling_mode in POOLING_MODES

        super().__init__()
        self.hparams = hparams

    def forward(self, tensor: torch.Tensor):
        """
        Input:
            tensor: (bsz, seq_len, emb_dim)

        output:
            pooled_tensor: (bsz, emb_dim)
        """
        if self.hparams.pooling_mode == "avg":
            return tensor.mean(1)
        elif self.hparams.pooling_mode == "max":
            return tensor.max(1)
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
