import argparse

import datasets
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase

from paoding.data.dataset import Dataset


class HuggingfaceDataset(Dataset):
    def __init__(
        self,
        hparams: argparse.Namespace,
        tokenizer: PreTrainedTokenizerBase,
        preprocess_and_save: bool = True,
        tokenize_separately: bool = False,
        *,
        dataset_name: str,
        subset_name: str,
        text_key: str,
        second_text_key: str = None,
        label_key: str,
        output_mode: str,
        num_labels: int = None,
        metric_names: list[str],
        metric_watch_mode: str,
    ):
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self._text_key = text_key
        self._second_text_key = second_text_key
        self._label_key = label_key
        self._output_mode = output_mode
        self._num_labels = num_labels
        self._metric_names = metric_names
        self._metric_watch_mode = metric_watch_mode

        super().__init__(
            hparams,
            tokenizer,
            preprocess_and_save=preprocess_and_save,
            tokenize_separately=tokenize_separately,
        )

    @property
    def text_key(self) -> str:
        return self._text_key

    @property
    def second_text_key(self) -> str:
        return self._second_text_key

    @property
    def label_key(self) -> str:
        return self._label_key

    @property
    def output_mode(self) -> str:
        return self._output_mode

    @property
    def num_labels(self) -> int:
        return self._num_labels

    @property
    def metric_names(self) -> list[str]:
        return self._metric_names

    @property
    def metric_watch_mode(self) -> str:
        return self._metric_watch_mode

    @property
    def hash_fields(self) -> str:
        return super().hash_fields + [self.dataset_name, self.subset_name]

    def load(self) -> DatasetDict:
        return datasets.load_dataset(self.dataset_name, self.subset_name)
