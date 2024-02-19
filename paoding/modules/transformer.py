import argparse
import logging

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    # AutoModelForPreTraining,
    # AutoModelForQuestionAnswering,
    # AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    # AutoModelForTokenClassification,
    AutoModelForCausalLM,
    # AutoModelForMaskedLM,
)
from transformers.models.auto.auto_factory import _get_model_class

from paoding.argument_parser import ArgumentParser


logger = logging.getLogger(__name__)


TASKS = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "causal-lm": AutoModelForCausalLM,
    # The below tasks haven't been tested. Uncomment and test when needed.
    # "question-answering": AutoModelForQuestionAnswering,
    # "pretraining": AutoModelForPreTraining,
    # "token-classification": AutoModelForTokenClassification,
    # "masked-lm": AutoModelForMaskedLM,
    # "summarization": AutoModelForSeq2SeqLM,
    # "translation": AutoModelForSeq2SeqLM,
}


class Transformer(torch.nn.Module):
    def __init__(self, hparams: argparse.Namespace, task: str, trainable=True, **config_kwargs):
        super().__init__()

        config_args = dict(config_kwargs)
        if task == "base":  # TODO: this might break models that don't support this flag
            config_args["add_pooling_layer"] = False
        self.config = AutoConfig.from_pretrained(
            hparams.transformer_model, revision=hparams.revision, **config_args
        )
        if hparams.random_init_transformer:
            self.model = _get_model_class(self.config, TASKS[task]._model_mapping)(self.config)
        else:
            self.model = TASKS[task].from_pretrained(
                hparams.transformer_model, revision=hparams.revision, config=self.config
            )

        if not trainable:
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self):
        return self.config.hidden_size

    def forward(self, *args, **kwargs):
        # `transformers` doesn't take bool masks which is crazy
        if kwargs.get("attention_mask") is not None:
            kwargs["attention_mask"] = kwargs["attention_mask"].float()
        return self.model(*args, **kwargs)

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--transformer_model",
            default=None,
            type=str,
            required=True,
            help="Model identifier from huggingface.co/models. Technically this could also be a"
            " local path, but untested, esp. the behavior when --random_init_transformer.",
        )
        parser.add_argument("--revision", default=None, type=str)
        parser.add_argument("--random_init_transformer", action="store_true")

        return parser
