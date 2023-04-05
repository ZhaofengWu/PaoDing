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
        dataset_name: str = None,
        subset_name: str = None,
        from_disk_path: str = None,
        train_split: str = None,
        dev_splits: str = None,
        test_splits: str = None,
        text_key: str,
        second_text_key: str = None,
        label_key: str = None,
        sort_key: str | tuple[str, str] = None,
        task: str,
        num_labels: int = None,
        metric_names: list[str],
        metric_to_watch: str = None,
        split_to_watch: str = None,
    ):
        self.from_disk = from_disk_path is not None
        if self.from_disk:
            assert dataset_name is None and subset_name is None
            self.from_disk_path = from_disk_path
        else:
            assert dataset_name is not None and subset_name is not None
            self.dataset_name = dataset_name
            self.subset_name = subset_name

        self._train_split = train_split
        self._dev_splits = dev_splits
        self._test_splits = test_splits
        self._text_key = text_key
        self._second_text_key = second_text_key
        self._label_key = label_key
        self._sort_key = sort_key
        self._task = task
        self._num_labels = num_labels
        self._metric_names = metric_names
        self._metric_to_watch = metric_to_watch
        self._split_to_watch = split_to_watch

        super().__init__(
            hparams,
            tokenizer,
            preprocess_and_save=preprocess_and_save,
            tokenize_separately=tokenize_separately,
        )

    @property
    def train_split(self) -> str:
        return self._train_split if self._train_split is not None else super().train_split

    @property
    def dev_splits(self) -> str:
        return self._dev_splits if self._dev_splits is not None else super().dev_splits

    @property
    def test_splits(self) -> str:
        return self._test_splits if self._test_splits is not None else super().test_splits

    @property
    def text_key(self) -> str:
        return self._text_key

    @property
    def second_text_key(self) -> str:
        return self._second_text_key

    @property
    def label_key(self) -> str:
        return self._label_key if self._label_key is not None else super().label_key

    @property
    def sort_key(self) -> str | tuple[str, str]:
        return self._sort_key if self._sort_key is not None else super().sort_key

    @property
    def task(self) -> str:
        return self._task

    @property
    def num_labels(self) -> int:
        return self._num_labels if self._num_labels is not None else super().num_labels

    @property
    def metric_names(self) -> list[str]:
        return self._metric_names

    @property
    def metric_to_watch(self) -> str:
        return (
            self._metric_to_watch if self._metric_to_watch is not None else super().metric_to_watch
        )

    @property
    def split_to_watch(self) -> str:
        return self._split_to_watch if self._split_to_watch is not None else super().split_to_watch

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
