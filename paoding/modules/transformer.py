import argparse
import logging

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)


logger = logging.getLogger(__name__)


TASKS = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "masked-lm": AutoModelForMaskedLM,
    "causal-lm": AutoModelForCausalLM,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


class Transformer(torch.nn.Module):
    def __init__(self, hparams: argparse.Namespace, task: str, trainable=True, **config_kwargs):
        super().__init__()

        config_args = dict(config_kwargs)
        if task == "base":  # TODO: this might break models that don't support this flag
            config_args["add_pooling_layer"] = False
        self.config = AutoConfig.from_pretrained(hparams.model_name_or_path, **config_args)
        self.model = TASKS[task].from_pretrained(hparams.model_name_or_path, config=self.config)
        if hparams.random_init_transformer:
            self.model = type(self.model)(self.config)

        if not trainable:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    def forward(self, *args, **kwargs):
        # `transformers` doesn't take bool masks which is crazy
        if kwargs.get("attention_mask") is not None:
            kwargs["attention_mask"] = kwargs["attention_mask"].float()
        return self.model(*args, **kwargs)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )
        parser.add_argument("--random_init_transformer", action="store_true")

        return parser
