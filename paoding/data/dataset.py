import argparse
import os
import pickle

import torch


class Dataset:
    def __init__(self, hparam: argparse.Namespace, cache_suffix=""):
        cache_file = self._cache_path(suffix=cache_suffix)
        if os.path.exists(cache_file):
            self.data = pickle.load(open(cache_file, "rb"))
            return

        # TODO: load data

        # pickle.dump(data, open(cache_file, "wb"))

    def __getitem__(self, split: str):
        pass  # TODO

    def _cache_path(self, suffix="") -> str:
        escape = lambda s: s.replace("/", "_")
        pass  # TODO
        # "name_{suffix}"

    @property
    def metric_to_watch(self):
        pass  # TODO

    @property
    def metric_watch_mode(self):
        pass  # TODO

    @property
    def output_mode(self):
        pass  # TODO

    @property
    def num_labels(self):
        pass  # TODO

    @property
    def sort_key(self):
        pass  # TODO

    def compute_metrics(self, todo):
        pass  # TODO

    @staticmethod
    def add_data_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--max_seq_length",
            default=None,
            type=int,
            help="The maximum sequence length after tokenization, for both source and target. "
            "Sequences longer than this will be truncated.",
        )
        pass  # TODO


class DatasetSplit(torch.utils.data.Dataset):
    def __len__(self):
        pass  # TODO

    def __getitem__(self, index: int):
        pass  # TODO


# TODO: support huggingface dataset
