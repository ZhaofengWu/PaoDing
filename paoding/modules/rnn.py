import argparse
import logging

import torch
from torch import nn

from paoding.argument_parser import ArgumentParser


logger = logging.getLogger(__name__)


class RNN(torch.nn.Module):
    def __init__(
        self,
        hparams: argparse.Namespace,
        input_dim: int,
        num_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.rnn = getattr(nn, hparams.rnn_model)(
            input_dim,
            hparams.rnn_dim,
            num_layers,
            batch_first=True,
            dropout=hparams.rnn_dropout,
            bidirectional=bidirectional,
        )

    def forward(self, input: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            hidden (torch.Tensor): (bsz, seq_len, input_dim)
            attention_mask (torch.Tensor): (bsz, seq_len)

        Returns:
            per_token_output (torch.Tensor): (bsz, seq_len, output_dim)
            final_output (torch.Tensor): (bsz, n_layers * n_directions * output_dim) (*2 if LSTM)
        """
        lens = attention_mask.sum(dim=-1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            input, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, hidden = self.rnn(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        if isinstance(hidden, tuple):
            hidden = torch.cat(hidden, dim=-1)
        return output, hidden.transpose(0, 1).reshape(hidden.shape[1], -1)

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument("--rnn_model", type=str, required=True, help="E.g. LSTM, GRU, etc.")
        # TODO: technically these aren't ideal when a model needs multiple RNNs.
        # But well, who uses RNNs today, let alone more than 1?
        # The benefit of this rather than requiring it as an init arg is that it's more modular.
        parser.add_argument("--rnn_dim", type=int, required=True)
        parser.add_argument("--rnn_dropout", default=0.0, type=float)

        return parser
