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
datasets.disable_caching()

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
            if self.resplit_source_split is not None:
                self.dataset_dict = self.resplit_dataset(self.dataset_dict)
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
            self.tokenize_separately,
            self.task,
            self.resplit_source_split,
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
    def resplit_source_split(self) -> str:
        return None

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
    def prompt_and_response_key(self) -> tuple[str, str]:
        """For reward modeling, a tuple of the keys in the example dictionary for the prompt and
        the response (pointwise) or the chosen response (pairwise)."""
        return None

    @property
    def rejected_response_key(self) -> str:
        """For pairwise reward modeling, the key in the example dictionary for the rejected response."""
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
        if self.task == "pairwise_rm":
            return ("input_ids_chosen", "input_ids_rejected")
        elif self.second_text_key is not None and self.tokenize_separately:
            return ("input_ids_1", "input_ids_2")
        return "input_ids"

    @property
    def task(self) -> str:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def num_labels(self) -> int:
        if self.task == "regression":
            return 1
        elif self.task == "pairwise_rm":
            return None
        else:
            raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def metric_names(self) -> list[str]:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def metric_to_watch(self) -> str:
        """A metric in `self.metric_names`, or `"loss"`."""
        if len(self.metric_names) == 1:
            return self.metric_names[0]
        else:
            raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def split_to_watch(self) -> str:
        # When there are multiple dev splits, you can specify which one to watch. By default, the
        # average performance across the dev splits is used.
        return None

    @property
    def metric_split_to_watch(self) -> str:
        # Matches the format in model.eval_epoch_end()
        return (
            f"{self.metric_to_watch}_{self.split_to_watch}"
            if self.split_to_watch
            else self.metric_to_watch
        )

    def load(self) -> DatasetDict:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def resplit_dataset(self, dataset_dict: DatasetDict) -> DatasetDict:
        # Some datasets don't have splits, and we need to split them ourselves
        assert len(self.dev_splits) == 1 and len(self.test_splits) == 1  # this can be supported
        train, dev, test = 8, 1, 1  # hardcoding for now

        intermediate = dataset_dict[self.resplit_source_split].train_test_split(
            train_size=train / (train + dev + test),
            test_size=(dev + test) / (train + dev + test),
        )
        dev_test = intermediate["test"].train_test_split(
            train_size=dev / (dev + test), test_size=test / (dev + test)
        )
        return DatasetDict(
            {
                self.train_split: intermediate["train"],
                self.dev_splits[0]: dev_test["train"],
                self.test_splits[0]: dev_test["test"],
            }
        )

    @property
    def tokenize_kwargs(self) -> dict[str, Any]:
        return dict(padding=False, truncation=True, max_length=self.hparams.max_length)

    def tokenize_ex(self, examples: dict[str, list]) -> dict[str, list]:
        if self.prompt_and_response_key is not None:
            prompt_and_response_keys_and_suffixes = [[self.prompt_and_response_key, ""]]
            if self.rejected_response_key is not None:
                assert self.task == "pairwise_rm"
                prompt_and_response_keys_and_suffixes[0][1] = "_chosen"
                prompt_and_response_keys_and_suffixes.append(
                    [self.rejected_response_key, "_rejected"]
                )
            output = {}
            for (prompt_key, response_key), suffix in prompt_and_response_keys_and_suffixes:
                prompt_key, response_key = self.prompt_and_response_key
                messages = [
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                    for prompt, response in zip(
                        examples[prompt_key], examples[response_key], strict=True
                    )
                ]
                single_output = self.tokenizer.apply_chat_template(
                    messages, return_dict=True, **self.tokenize_kwargs
                )
                output.update({f"{k}{suffix}": v for k, v in single_output.items()})
            return output

        if self.tokenize_separately:
            output = {}
            for i, text in enumerate((examples[self.text_key], examples[self.second_text_key])):
                single_output = self.tokenizer(text, **self.tokenize_kwargs)
                output.update({f"{k}_{i + 1}": v for k, v in single_output.items()})
            return output
        else:
            return self.tokenizer(
                examples[self.text_key],
                text_pair=(
                    examples[self.second_text_key] if self.second_text_key is not None else None
                ),
                **self.tokenize_kwargs,
            )

    def tokenize_seq2seq(self, example: dict[str, Any]) -> dict[str, Any]:
        # Reference: https://github.com/huggingface/trl/blob/79686e1ac701b1f5e3709a65efa8f13363bcde06/trl/trainer/dpo_trainer.py#L678-L726
        assert not self.tokenize_separately

        prompt = example[self.text_key]
        answer = example[self.label_key]

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        full_input_ids = full_tokenized["input_ids"]
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # in the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        # prompt: a b c d
        # full  : a b c d' e f g
        common_prompt_len = len(prompt_input_ids)
        if prompt_input_ids != full_input_ids[:common_prompt_len]:
            common_prompt_len -= 1
        assert prompt_input_ids[:common_prompt_len] == full_input_ids[:common_prompt_len]

        assert self.tokenizer.eos_token_id is not None
        full_input_ids = full_input_ids + [self.tokenizer.eos_token_id]
        attention_mask = full_tokenized["attention_mask"] + [1]
        if self.tokenizer.bos_token_id is not None:
            full_input_ids = [self.tokenizer.bos_token_id] + full_input_ids
            attention_mask = [1] + attention_mask
            common_prompt_len += 1

        return {
            "input_ids": full_input_ids,
            "attention_mask": attention_mask,
            "prompt_len": common_prompt_len,
        }

    def tokenize(self, dataset_dict: DatasetDict, map_kwargs: dict = None) -> DatasetDict:
        return DatasetDict(  # reimplementing DatasetDict.map to provide `split`
            {
                split: dataset.map(
                    self.tokenize_ex if self.task != "seq2seq" else self.tokenize_seq2seq,
                    batched=self.task != "seq2seq",  # the logic is too complicated for batching
                    num_proc=4,
                    **(map_kwargs if map_kwargs is not None else {}),
                )
                for split, dataset in dataset_dict.items()
            }
        )

    def remap_ex_causal_lm(self, example: dict[str, Any], suffix="") -> dict[str, Any]:
        return {
            f"input_ids{suffix}": example[f"input_ids{suffix}"][:-1],
            f"attention_mask{suffix}": example[f"attention_mask{suffix}"][:-1],
            f"{self.label_key}{suffix}": example[f"input_ids{suffix}"][1:],
            f"{self.label_mask_key}{suffix}": example[f"attention_mask{suffix}"][1:],
        }

    def remap_ex_seq2seq(self, example: dict[str, Any], suffix="") -> dict[str, Any]:
        prompt_len = example["prompt_len"]
        return {
            f"input_ids{suffix}": example[f"input_ids{suffix}"][:-1],
            f"attention_mask{suffix}": example[f"attention_mask{suffix}"][:-1],
            f"{self.label_key}{suffix}": example[f"input_ids{suffix}"][prompt_len:],
            f"{self.label_mask_key}{suffix}": example[f"attention_mask{suffix}"][prompt_len:],
        }

    def remap_ex_masked_lm(self, example: dict[str, Any], suffix="") -> dict[str, Any]:
        return {
            f"{self.label_key}{suffix}": example[f"input_ids{suffix}"],
            f"{self.label_mask_key}{suffix}": example[f"attention_mask{suffix}"],
        }

    def prepare_input_for_lm(self, dataset_dict: DatasetDict) -> DatasetDict:
        remap_ex = {
            "causal_lm": self.remap_ex_causal_lm,
            "seq2seq": self.remap_ex_seq2seq,
            "masked_lm": self.remap_ex_masked_lm,
        }[self.task]
        return DatasetDict(
            {
                k: d.map(
                    lambda example: (
                        remap_ex(example)
                        if not self.tokenize_separately
                        else {**remap_ex(example, "_1"), **remap_ex(example, "_2")}
                    ),
                    batched=False,
                    num_proc=4,
                )
                for k, d in dataset_dict.items()
            }
        )

    def preprocess(self, dataset_dict: DatasetDict, map_kwargs: dict = None) -> DatasetDict:
        dataset_dict = self.tokenize(dataset_dict, map_kwargs=map_kwargs)

        # Rename validation -> dev
        if "validation" in dataset_dict and "dev" not in dataset_dict:
            dataset_dict["dev"] = dataset_dict["validation"]
            del dataset_dict["validation"]

        if self.task in {"causal_lm", "seq2seq", "masked_lm"}:
            dataset_dict = self.prepare_input_for_lm(dataset_dict)

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
        if (
            shuffle and not self.hparams.no_sort
        ):  # TODO: think about this -- sorting by length will be faster for validation too; maybe when validating but not testing? But sorting messes up prediction logging
            lens = [0] * len(dataset_split)
            for k in self.sort_key if not isinstance(self.sort_key, str) else [self.sort_key]:
                for i, v in enumerate(dataset_split[k]):  # TODO: this is a bit slow
                    # LengthGroupedSampler sorts from longest to shortest; we want the reverse
                    lens[i] -= len(v)
            if self.hparams.gpus <= 1:
                sampler = LengthGroupedSampler(batch_size, lengths=lens)
            else:
                sampler = DistributedLengthGroupedSampler(batch_size, lengths=lens)
            shuffle = False  # can't specify both shuffle and sampler

        batch_info = self.batch_info(split)
        dataloader = DataLoader(
            dataset_split,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_dataloader_workers(),
            collate_fn=lambda batch: self.after_collation(
                collate_fn(self.before_collation(batch), batch_info, self.tokenizer.padding_side)
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
        if self.task == "pairwise_rm":
            new_batch_info = {}
            for response_desc in ("chosen", "rejected"):
                new_batch_info.update({f"{k}_{response_desc}": v for k, v in batch_info.items()})
            batch_info = new_batch_info
        if self.tokenize_separately:
            new_batch_info = {}
            for i in (1, 2):
                new_batch_info.update({f"{k}_{i}": v for k, v in batch_info.items()})
            batch_info = new_batch_info

        # label
        if self.task != "pairwise_rm":
            label_dtype = (
                torch.float if self.task in {"regression", "multi_regression"} else torch.long
            )
            label_pad = None
            if self.task in {"causal_lm", "seq2seq", "masked_lm"}:
                label_pad = (
                    self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                )
            batch_info.update({self.label_key: (label_dtype, label_pad)})
            if self.task in {"causal_lm", "seq2seq", "masked_lm"}:
                if self.tokenize_separately:
                    del batch_info[self.label_key]
                    batch_info.update(
                        {
                            f"{self.label_key}_1": (label_dtype, label_pad),
                            f"{self.label_key}_2": (label_dtype, label_pad),
                            f"{self.label_mask_key}_1": (torch.bool, False),
                            f"{self.label_mask_key}_2": (torch.bool, False),
                        }
                    )
                else:
                    batch_info.update({self.label_mask_key: (torch.bool, False)})

        if self.task == "seq2seq":
            batch_info.update({"prompt_len": (torch.long, 0)})

        return batch_info

    def before_collation(self, batch: list[dict[str, list]]) -> list[dict[str, list]]:
        """
        Allows subclasses to have a chance to modify the batch
        """
        return batch

    def after_collation(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.task == "masked_lm":
            # Adapted from https://github.com/huggingface/transformers/blob/v4.23.0/src/transformers/data/data_collator.py#L748-L779
            input_ids = batch["input_ids"]
            shape = input_ids.shape
            labels = batch[self.label_key]
            label_mask = batch[self.label_mask_key]

            # We sample a few tokens in each sequence for MLM training
            probability_matrix = torch.full(shape, self.hparams.mlm_probability)
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            label_mask[~masked_indices] = False  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(shape, 0.8)).bool() & masked_indices
            input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.mask_token
            )

            # 10% of the time, we replace masked input tokens with random word
            indices_random = (
                torch.bernoulli(torch.full(shape, 0.5)).bool() & masked_indices & ~indices_replaced
            )
            random_words = torch.randint(len(self.tokenizer), shape, dtype=torch.long)
            input_ids[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged

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
        parser.add_argument(
            "--mlm_probability",
            default=0.15,  # same as roberta
            type=float,
            help="Only used when task is masked_lm. The probability with which to (randomly) mask.",
        )
