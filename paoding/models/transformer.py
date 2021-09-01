import logging

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
)

from paoding.models.base_model import BaseModel


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


class Transformer(BaseModel):
    def __init__(self, hparams, **config_kwargs):
        super().__init__(hparams)

        self.config = AutoConfig.from_pretrained(
            self.hparams.model_name_or_path,
            **(
                {"num_labels": self.dataset.num_labels}
                if self.dataset.num_labels is not None
                else {}
            ),
            **config_kwargs,
        )
        self.model = TASKS[self.hparams.task].from_pretrained(
            self.hparams.model_name_or_path, config=self.config
        )

    def forward(self, *args, **kwargs):
        if "attention_mask" in kwargs:  # `transformers` doesn't take bool masks which is crazy
            kwargs["attention_mask"] = kwargs["attention_mask"].float()
        return self.model(*args, **kwargs)

    def _step(self, batch):
        input = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        if "token_type_ids" in batch and self.config.model_type != "distilbert":
            input["token_type_ids"] = batch["token_type_ids"]
        return self(**input)

    @staticmethod
    def add_args(parser):
        BaseModel.add_args(parser)

        parser.add_argument(
            "--task",
            default=None,
            type=str,
            required=True,
            choices=TASKS.keys(),
            help="The task of the model.",
        )
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )

        return parser
