import argparse
import os
from typing import Callable

from datasets import DatasetDict, load_dataset

from paoding.data.dataset import Dataset


class LocalDataset(Dataset):
    def __init__(
        self,
        hparams: argparse.Namespace,
        split_filename: Callable[[str], str],
        *load_args,
        **load_kwargs,
    ):
        self.split_filename = split_filename
        self.load_args = load_args
        self.load_kwargs = load_kwargs
        super().__init__(hparams)

    def load(self) -> DatasetDict:
        dataset_dict = load_dataset(
            *self.load_args,
            data_files={
                split: os.path.join(self.hparams.data_dir, self.split_filename(split))
                for split in [self.train_split] + self.dev_splits + self.test_splits
            },
            **self.load_kwargs,
        )
        assert isinstance(dataset_dict, DatasetDict)
        return dataset_dict

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        Dataset.add_args(parser)
        parser.add_argument("--data_dir", required=True, type=str)
