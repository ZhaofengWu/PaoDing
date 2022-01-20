import argparse
from typing import Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from paoding.models.model import Model
from paoding.modules.transformer import Transformer


class TransformerModel(Model):
    def __init__(
        self, hparams: Union[argparse.Namespace, dict], task: str, trainable=True, **config_kwargs
    ):
        super().__init__(hparams)
        self.transformer = Transformer(hparams, task, trainable=trainable, **config_kwargs)

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)

    def forward(self, batch: dict[str, torch.Tensor]):
        return self.transformer(**{k: v for k, v in batch.items() if k != self.dataset.label_key})

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        Model.add_args(parser)
        Transformer.add_args(parser)