import argparse
from typing import Union

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
        dataset_name: str = None,
        subset_name: str = None,
        from_disk_path: str = None,
        text_key: str,
        second_text_key: str = None,
        label_key: str,
        sort_key: Union[str, tuple[str]] = None,
        output_mode: str,
        num_labels: int = None,
        metric_names: list[str],
        metric_watch_mode: str,
    ):
        self.from_disk = from_disk_path is not None
        if self.from_disk:
            assert dataset_name is None and subset_name is None
            self.from_disk_path = from_disk_path
        else:
            assert dataset_name is not None and subset_name is not None
            self.dataset_name = dataset_name
            self.subset_name = subset_name

        self._text_key = text_key
        self._second_text_key = second_text_key
        self._label_key = label_key
        self._sort_key = sort_key
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
    def sort_key(self) -> Union[str, tuple[str]]:
        return self._sort_key if self._sort_key is not None else super().sort_key

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
        return super().hash_fields + (
            [self.from_disk_path] if self.from_disk else [self.dataset_name, self.subset_name]
        )

    def load(self) -> DatasetDict:
        if self.from_disk:
            return datasets.load_from_disk(self.from_disk_path)
        else:
            return datasets.load_dataset(self.dataset_name, self.subset_name)
