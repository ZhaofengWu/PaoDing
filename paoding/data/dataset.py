import argparse
from collections.abc import ItemsView
import hashlib
import math
import os
import random
import torch
from typing import Any

import datasets
from datasets import DatasetDict, Dataset as HFDataset
from torch.utils.data.dataloader import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import LengthGroupedSampler, DistributedLengthGroupedSampler

from paoding.argument_parser import ArgumentParser
from paoding.data.collator import collate_fn, BATCH_INFO
from paoding.utils import get_logger

# Sometimes we want to change the implementation of methods, etc., which cache ignores.
# We maintain our own cache so this is not very useful anyway.
datasets.set_caching_enabled(False)

logger = get_logger(__name__)


class Dataset:
    """
    An abstract class representing a dataset (using the HuggingFace datasets), relevant properties,
    and a tokenizer.

    High level logic:
    1. self.load() loads a DatasetDict in the format:
        {
            split:
            [{text_key: xxx, (second_text_key: yyy), label_key: l}] for split in self.all_splits
        }
    2. self.preprocess() tokenizes the text(s), and now each example has keys:
        (
            text_key, (second_text_key), label, input_ids, attention_mask, (label_mask),
            (other tokenizer outputs such as token_type_ids)
        )
        Alternatively, if tokenize_separately is True, the tokenizer-added fields will be separate
        to each text.
    3. self.collate_fn() selects only the fields in batch_info to assemble into a batch. By default,
        this is just the tokenizer-added fields plus the label (and label mask for LM).
    """

    def __init__(
        self,
        hparams: argparse.Namespace,
        tokenizer: PreTrainedTokenizerBase,
        preprocess_and_save: bool = True,
        tokenize_separately: bool = False,
    ):
        """
        Input:
            preprocess_and_save: sometimes a Dataset is used as an intermediate processing step,
                in which case no preprocessing and persistence may be necessary.
            tokenize_separately: if False, both sentences, if there are two, in an example will be
                passed to the tokenizer together. Otherwise, the sentences are separately tokenized,
                and the keys are suffixed with the (1-based) index.
        """
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.tokenize_separately = tokenize_separately
        if preprocess_and_save and os.path.exists(self.cache_path):
            logger.info(f"Reusing cache at {self.cache_path}")
            self.dataset_dict = DatasetDict.load_from_disk(self.cache_path)
        else:
            self.dataset_dict = self.load()
            if preprocess_and_save:
                self.dataset_dict = self.preprocess(self.dataset_dict)
                logger.info(f"Saving dataset cache at {self.cache_path}")
                self.dataset_dict.save_to_disk(self.cache_path)

        self.hparams.subsample_training_ratio = getattr(
            self.hparams, "subsample_training_ratio", None
        )
        if self.hparams.subsample_training_ratio is not None:
            orig_size = len(self.dataset_dict[self.train_split])
            subsampled_size = int(math.floor(orig_size * self.hparams.subsample_training_ratio))
            indices = random.sample(range(orig_size), subsampled_size)
            self.dataset_dict[self.train_split] = self.dataset_dict[self.train_split].select(
                indices
            )

    def items(self) -> ItemsView:
        return self.dataset_dict.items()

    def __getitem__(self, key: str) -> HFDataset:
        return self.dataset_dict[key]

    @property
    def hash_fields(self) -> list[Any]:
        """For cache purpose"""
        return [
            self.__class__,
            self.hparams.seed,
            self.tokenizer.__repr__(),
            self.hparams.max_length,
        ]

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
    def all_splits(self) -> list[str]:
        return [self.train_split] + self.dev_splits + self.test_splits

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
        return "label"

    @property
    def label_mask_key(self) -> str:
        """The key in the example dictionary for the label mask, for tasks such as LM."""
        return "label_mask"

    @property
    def sort_key(self) -> str | tuple[str, str]:
        return (
            "input_ids"
            if self.second_text_key is None or not self.tokenize_separately
            else ("input_ids_1", "input_ids_2")
        )

    @property
    def task(self) -> str:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def num_labels(self) -> int:
        if self.task == "regression":
            return 1
        else:
            raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def metric_names(self) -> list[str]:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def metric_to_watch(self) -> str:
        if len(self.metric_names) == 1:
            return self.metric_names[0]
        else:
            raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def load(self) -> DatasetDict:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def tokenize_kwargs(self) -> dict[str, Any]:
        return dict(padding=False, truncation=True, max_length=self.hparams.max_length)

    def tokenize(self, examples: dict[str, list], split: str) -> dict[str, list]:
        if self.tokenize_separately:
            output = {}
            for i, text in enumerate((examples[self.text_key], examples[self.second_text_key])):
                single_output = self.tokenizer(text, **self.tokenize_kwargs)
                output.update({f"{k}_{i + 1}": v for k, v in single_output.items()})
            return output
        else:
            return self.tokenizer(
                examples[self.text_key],
                text_pair=examples[self.second_text_key]
                if self.second_text_key is not None
                else None,
                **self.tokenize_kwargs,
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

        if self.task == "causal_lm":
            dataset_dict = DatasetDict(
                {
                    k: d.map(
                        lambda examples: {
                            "input_ids": examples["input_ids"][:-1],
                            "attention_mask": examples["attention_mask"][:-1],
                            self.label_key: examples["input_ids"][1:],
                            self.label_mask_key: examples["attention_mask"][1:],
                        },
                        batched=False,
                        num_proc=4,
                    )
                    for k, d in dataset_dict.items()
                }
            )

        return dataset_dict

    def num_dataloader_workers(self) -> int:
        # Modified from https://github.com/pytorch/pytorch/blob/7c98e70d44abc7a1aead68b6ea6c8adc8c554db5/torch/utils/data/dataloader.py#L482
        if self.hparams.debug:
            return 0
        if hasattr(os, "sched_getaffinity"):
            try:
                return len(os.sched_getaffinity(0))
            except Exception:
                pass
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            return cpu_count
        return 2

    def dataloader(self, split: str, batch_size: int, shuffle=False) -> DataLoader:
        dataset_split = self.dataset_dict[split]
        sampler = None
        if shuffle and not self.hparams.no_sort:  # TODO: think about this -- sorting by length will be faster for validation too; maybe when validating but not testing? But sorting messes up prediction logging
            lens = [0] * len(dataset_split)
            for k in self.sort_key if not isinstance(self.sort_key, str) else [self.sort_key]:
                for i, v in enumerate(dataset_split[k]):  # TODO: this is a bit slow
                    # LengthGroupedSampler sorts from longest to shortest; we want the reverse
                    lens[i] -= len(v)
            if self.hparams.gpus <= 1:
                sampler = LengthGroupedSampler(batch_size, lengths=lens)
            else:
                sampler = DistributedLengthGroupedSampler(batch_size, lengths=lens)
            shuffle = False  # can't specify both shuffle and smapler

        batch_info = self.batch_info(split)
        dataloader = DataLoader(
            dataset_split,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_dataloader_workers(),
            collate_fn=lambda batch: collate_fn(
                self.before_collation(batch), batch_info, self.tokenizer.padding_side
            ),
            pin_memory=True,
        )
        return dataloader

    def batch_info(self, split: str) -> BATCH_INFO:
        """
        Specifies the fields to be included in a batch, and their types and padding values. A None
        padding value means no padding, and collation will complain if the batch contains examples
        with different shapes.
        """
        known_tokenizer_fields = {  # fields we know how to handle
            "input_ids": (
                torch.long,
                # Separate check for tokenizers that don't have a pad token id (stupid gpt2).
                # This is a little dangerous, but should be ok if we rely on attention_mask.
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
            ),
            "attention_mask": (torch.bool, False),
            "token_type_ids": (torch.long, self.tokenizer.pad_token_type_id),
        }
        if any(k not in known_tokenizer_fields for k in self.tokenizer.model_input_names):
            missing = set(self.tokenizer.model_input_names) - set(known_tokenizer_fields.keys())
            raise KeyError(f"Don't know how to handle batch keys {missing}.")

        batch_info = {
            name: known_tokenizer_fields[name] for name in self.tokenizer.model_input_names
        }
        if self.tokenize_separately:
            new_batch_info = {}
            for i in (1, 2):
                new_batch_info.update({f"{k}_{i}": v for k, v in batch_info.items()})
            batch_info = new_batch_info

        label_dtype = torch.float if self.task == "regression" else torch.long
        # TODO: this won't work for gpt2 now. See https://github.com/Lightning-AI/metrics/issues/54
        # We could default to using 0 like above, but `torchmetrics` doesn't support masks yet,
        # which makes it dangerous.
        assert self.tokenizer.pad_token_id is not None
        label_pad = self.tokenizer.pad_token_id if self.task == "causal_lm" else None
        batch_info.update({self.label_key: (label_dtype, label_pad)})
        if self.task == "causal_lm":
            batch_info.update({self.label_mask_key: (torch.bool, False)})
        return batch_info

    def before_collation(self, batch: list[dict[str, list]]) -> list[dict[str, list]]:
        """
        Allows subclasses to have a chance to modify the batch
        """
        return batch

    @staticmethod
    def add_args(parser: ArgumentParser):
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
            help="The maximum sequence length after tokenization. Sequences longer than this will"
            " be truncated.",
        )
        parser.add_argument(
            "--subsample_training_ratio",
            default=None,
            type=float,
            help="If specified, the ratio at which to random subsample the training set.",
        )
        parser.add_argument(
            "--no_sort", action="store_true", help="Disable (approximately) sorting by length."
        )
