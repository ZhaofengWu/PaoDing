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
    AutoModelWithLMHead,
)


logger = logging.getLogger(__name__)


TASKS = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


class Transformer(torch.nn.module):
    def __init__(self, hparams, task, trainable=True, **config_kwargs):
        super().__init__()

        config_args = dict(config_kwargs)
        if task == "base":  # TODO: this might break models that don't support this flag
            config_args["add_pooling_layer"] = False
        self.config = AutoConfig.from_pretrained(hparams.model_name_or_path, **config_args)
        self.model = TASKS[task].from_pretrained(hparams.model_name_or_path, config=self.config)

        if not trainable:  # TODO: support this
            assert task == "base", "No support for freezing the backbone for headed tasks yet"
        self.trainable = trainable

    def forward(self, *args, **kwargs):
        if "attention_mask" in kwargs:  # `transformers` doesn't take bool masks which is crazy
            kwargs["attention_mask"] = kwargs["attention_mask"].float()
        with torch.set_grad_enabled(self.trainable):
            return self.model(*args, **kwargs)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )

        return parser
