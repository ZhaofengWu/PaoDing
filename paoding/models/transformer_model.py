import argparse
import os

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from paoding.argument_parser import ArgumentParser
from paoding.models.model import Model
from paoding.modules.transformer import Transformer


class TransformerModel(Model):
    def __init__(
        self, hparams: argparse.Namespace | dict, task: str, trainable=True, **config_kwargs
    ):
        super().__init__(hparams)
        if "num_labels" not in config_kwargs and task == "sequence-classification":
            config_kwargs["num_labels"] = self.dataset.num_labels
        self.transformer = Transformer(self.hparams, task, trainable=trainable, **config_kwargs)

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        token = os.environ.get("HF_TOKEN")
        return AutoTokenizer.from_pretrained(self.hparams.transformer_model, token=token)

    def forward(self, batch: dict[str, torch.Tensor]):
        return self.transformer(
            **{
                k: v
                for k, v in batch.items()
                if k not in {self.dataset.label_key, self.dataset.label_mask_key}
            }
        )

    @staticmethod
    def add_args(parser: ArgumentParser):
        Model.add_args(parser)
        Transformer.add_args(parser)
