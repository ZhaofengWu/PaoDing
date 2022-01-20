import argparse
from collections.abc import ItemsView
import hashlib
import os
from typing import Any

import datasets
from datasets import DatasetDict, Dataset as HFDataset
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import LengthGroupedSampler

from paoding.data.collator import collate_fn, PAD_TYPE
from paoding.utils import get_logger

# Sometimes we want to change the implementation of methods, etc., which cache ignores.
# We maintain our own cache so this is not very useful anyway.
datasets.set_caching_enabled(False)

logger = get_logger(__name__)


class Dataset:
    """
    An abstract class representing a dataset (using the HuggingFace datasets), relevant properties,
    and a tokenizer.
    """

    def __init__(
        self,
        hparams: argparse.Namespace,
        tokenizer: PreTrainedTokenizerBase,
        preprocess_and_save: bool = True,
    ):
        """
        Input:
            preprocess_and_save: sometimes a Dataset is used as an intermediate processing step,
                in which case no preprocessing and persistence may be necessary.
        """
        self.hparams = hparams
        self.tokenizer = tokenizer
        if preprocess_and_save:
            if os.path.exists(self.cache_path):
                logger.info(f"Reusing cache at {self.cache_path}")
                self.dataset_dict = DatasetDict.load_from_disk(self.cache_path)
                return

        self.dataset_dict = self.load()
        if preprocess_and_save:
            self.dataset_dict = self.preprocess(self.dataset_dict)
            logger.info(f"Saving dataset cache at {self.cache_path}")
            self.dataset_dict.save_to_disk(self.cache_path)

    def __getitem__(self, key: str) -> HFDataset:
        return self.dataset_dict[key]

    @property
    def hash_fields(self) -> list[Any]:
        """For cache purpose"""
        return [self.hparams.seed, self.tokenizer.__repr__(), self.hparams.max_length]

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

    def preprocess(self, dataset_dict: DatasetDict, map_kwargs: dict = None) -> DatasetDict:
        dataset_dict = DatasetDict(  # reimplementing DatasetDict.map to provide `split`
            {
                split: dataset.map(
                    lambda examples: self.tokenize(examples, split),
                    batched=True,
                    num_proc=4,
                    **(map_kwargs if map_kwargs is not None else {}),
                )
                for split, dataset in dataset_dict.items()
            }
        )

        # Rename validation -> dev
        if "validation" in dataset_dict and "dev" not in dataset_dict:
            dataset_dict["dev"] = dataset_dict["validation"]
            del dataset_dict["validation"]

        return dataset_dict

    def items(self) -> ItemsView:
        return self.dataset_dict.items()

    def dataloader(self, split: str, batch_size: int, shuffle=False) -> DataLoader:
        dataset_split = self.dataset_dict[split]
        if shuffle:
            # LengthGroupedSampler sorts from longest to shortest; we want the reverse
            lens = [-len(ids) for ids in dataset_split[self.sort_key]]
            if self.hparams.gpus <= 1:
                sampler = LengthGroupedSampler(None, batch_size, lengths=lens)
            else:
                # TODO: support this when https://github.com/huggingface/transformers/commit/1b74af76b7e5c259d1470dec9d8d68c303dea5db is released
                # and also remove the None from above
                raise NotImplementedError()
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
                self.before_collation(batch),
                self.label_key,
                pad_token_map,
                self.tokenizer.padding_side,
                self.output_mode,
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

    def before_collation(self, batch: list[dict[str, list]]) -> list[dict[str, list]]:
        """
        Allows subclasses to have a chance to modify the batch
        """
        return batch

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
