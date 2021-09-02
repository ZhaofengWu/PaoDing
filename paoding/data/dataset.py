import argparse
import hashlib
import os
import pickle

from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase


class Dataset:
    """
    An abstract class representing a dataset (using the HuggingFace datasets), a tokenizer, and
    metrics.
    """

    def __init__(self, hparam: argparse.Namespace):
        self.hparam = hparam
        if os.path.exists(self.cache_path):
            self.dataset_dict = pickle.load(open(self.cache_path, "rb"))
            return

        self.tokenizer = self.setup_tokenizer()

        self.dataset_dict = self.load()
        self.dataset_dict = self.preprocess(self.dataset_dict)
        pickle.dump(self.dataset_dict, open(self.cache_path, "wb"))

    def __getitem__(self, key):
        return self.dataset_dict[key]

    @property
    def cache_path(self) -> str:
        hash = lambda s: hashlib.md5(s.encode("utf-8")).hexdigest()
        return os.path.join(
            self.hparam.output_dir,
            f"{self.__class__.__name__}_{hash(self.tokenizer.__repr__)}.datacache",
        )

    @property
    def text_key(self) -> str:
        """The key in the example dictionary for the main text."""
        return "text"

    @property
    def text_pair_key(self) -> str:
        """For text pairs, the key in the example dictionary for the second text."""
        return None

    @property
    def sort_key(self) -> str:
        return self.text_key

    @property
    def metrics(self) -> list[str]:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @property
    def metric_to_watch(self) -> str:
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

    def preprocess(self, dataset_dict: DatasetDict) -> DatasetDict:
        tokenization_fn = lambda examples: self.tokenizer(
            examples[self.text_key],
            text_pair=examples[self.text_pair_key] if self.text_pair_key is not None else None,
            padding=False,  # we control this in the collator
            truncation=True,
            max_length=self.tokenizer.model_max_len,
        )
        self.dataset_dict = dataset_dict.map(tokenization_fn, batched=True, num_proc=4)

        # Rename validation -> dev
        if "validation" in dataset_dict and "dev" not in dataset_dict:
            dataset_dict["dev"] = dataset_dict["validation"]
            del dataset_dict["validation"]

        return dataset_dict

    def setup_tokenizer(self) -> PreTrainedTokenizerBase:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def compute_metrics(self, todo):
        pass  # TODO

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument(  # subclasses should use this when instantiating the tokenizer
            "--max_length",
            default=None,
            type=int,
            help="The maximum sequence length after tokenization, for both source and target. "
            "Sequences longer than this will be truncated.",
        )
