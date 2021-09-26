import argparse
from collections.abc import ItemsView
import hashlib
import os
from typing import Any

import datasets
from datasets import DatasetDict, Dataset as HFDataset
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizerBase

from paoding.data.collator import collate_fn, PAD_TYPE
from paoding.data.sortish_sampler import make_sortish_sampler

# Sometimes we want to change the implementation of methods, etc., which cache ignores.
# We maintain our own cache so this is not very useful anyway.
datasets.set_caching_enabled(False)

class Dataset:
    """
    An abstract class representing a dataset (using the HuggingFace datasets), relevant properties,
    and a tokenizer.
    """

    def __init__(self, hparams: argparse.Namespace, preprocess_and_save=True):
        """
        Input:
            preprocess_and_save: sometimes a Dataset is used as an intermediate processing step,
                in which case no preprocessing and persistence may be necessary.
        """
        self.hparams = hparams
        if preprocess_and_save:
            self.tokenizer = self.setup_tokenizer()
            if os.path.exists(self.cache_path):
                self.dataset_dict = DatasetDict.load_from_disk(self.cache_path)
                return

        self.dataset_dict = self.load()
        if preprocess_and_save:
            self.dataset_dict = self.preprocess(self.dataset_dict)
            self.dataset_dict.save_to_disk(self.cache_path)

    def __getitem__(self, key: str) -> HFDataset:
        return self.dataset_dict[key]

    @property
    def hash_fields(self) -> list[Any]:
        """For cache purpose"""
        return [self.tokenizer.__repr__]

    @property
    def cache_path(self) -> str:
        hash = lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()
        hash_fields = "".join([str(f) for f in self.hash_fields])
        return os.path.join(
            self.hparams.data_dir,
            f"{self.__class__.__name__}_{hash(hash_fields)}.datacache",
        )

    @property
    def train_split(self) -> str:
        return "train"

    @property
    def dev_splits(self) -> list[str]:
        return ["dev"]

    @property
    def test_splits(self) -> list[str]:
        return ["test"]

    @property
    def text_key(self) -> str:
        """The key in the example dictionary for the main text."""
        return "text"

    @property
    def second_text_key(self) -> str:
        """For text pairs, the key in the example dictionary for the second text."""
        return None

    @property
    def label_key(self) -> str:
        """The key in the example dictionary for the label."""
        return "label"

    @property
    def sort_key(self) -> str:
        return self.text_key

    @property
    def metric_names(self) -> list[str]:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def metric_to_watch(self) -> str:
        if len(self.metric_names) == 1:
            return self.metric_names[0]
        else:
            raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def metric_watch_mode(self) -> str:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def output_mode(self) -> str:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def num_labels(self) -> int:
        if self.output_mode == "classification":
            raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")
        return None

    def load(self) -> DatasetDict:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def tokenize(self, examples: dict[str, list], split: str) -> dict[str, list]:
        return self.tokenizer(
            examples[self.text_key],
            text_pair=examples[self.second_text_key] if self.second_text_key is not None else None,
            padding=False,  # we control this in the collator
            truncation=True,
            max_length=self.hparams.max_length,
        )

    def preprocess(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = DatasetDict(  # reimplementing DatasetDict.map to provide `split`
            {
                split: dataset.map(
                    lambda examples: self.tokenize(examples, split),
                    batched=True,
                    num_proc=4,
                )
                for split, dataset in dataset_dict.items()
            }
        )

        # Rename validation -> dev
        if "validation" in dataset_dict and "dev" not in dataset_dict:
            dataset_dict["dev"] = dataset_dict["validation"]
            del dataset_dict["validation"]

        return dataset_dict

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def items(self) -> ItemsView:
        return self.dataset_dict.items()

    def dataloader(self, split: str, batch_size: int, shuffle=False) -> DataLoader:
        dataset_split = self.dataset_dict[split]
        lens = [len(ids) for ids in dataset_split[self.sort_key]]
        if shuffle:
            sampler = make_sortish_sampler(
                lens, batch_size, distributed=self.hparams.gpus > 1, perturb=True
            )
        else:
            sampler = None
        pad_token_map = self.pad_token_map(split)
        assert all(pad is not None for pad in pad_token_map.values())
        dataloader = DataLoader(
            dataset_split,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=2,
            collate_fn=lambda batch: collate_fn(
                batch, self.label_key, pad_token_map, self.tokenizer.padding_side, self.output_mode
            ),
            pin_memory=True,
        )
        return dataloader

    def pad_token_map(self, split: str) -> dict[str, PAD_TYPE]:
        """
        Specifies the padding for each key. Only keys including in this map plus the label will be
        included in the batch.
        """
        return {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": False,
            "token_type_ids": self.tokenizer.pad_token_type_id,
        }

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--data_dir",
            required=True,
            type=str,
            help="The location to data for local datasets, and the directory where the data cache"
            " will be stored for all datasets.",
        )
        parser.add_argument(
            "--max_length",
            default=None,
            type=int,
            help="The maximum sequence length after tokenization, for both source and target."
            " Sequences longer than this will be truncated.",
        )
