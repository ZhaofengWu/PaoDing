import argparse
import os

from datasets import DatasetDict, load_dataset

from paoding.data.dataset import Dataset


class LocalDataset(Dataset):
    @property
    def load_dataset_args(self) -> tuple[list, dict]:
        # e.g. ["csv"], {"delimiter": "=", "column_names": ["text", "label"]}
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def split_filename(self, split) -> str:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def load(self) -> DatasetDict:
        args, kwargs = self.load_dataset_args
        dataset_dict = load_dataset(
            *args,
            data_files={
                split: os.path.join(self.hparams, self.split_filename(split))
                for split in ["train", "dev", "test"]
            },
            **kwargs,
        )
        assert isinstance(dataset_dict, DatasetDict)
        return dataset_dict

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        Dataset.add_args(parser)
        parser.add_argument("--data_dir", required=True, type=str)
