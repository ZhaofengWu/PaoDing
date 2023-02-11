import argparse
import os
import pickle
from typing import Callable

from datasets import Dataset as HFDataset, DatasetDict, load_dataset
from datasets.packaged_modules import _EXTENSION_TO_MODULE
from transformers import PreTrainedTokenizerBase

DATASETS_SUPPORTED_FORMATS = _EXTENSION_TO_MODULE.keys()

from paoding.data.dataset import Dataset


class LocalDataset(Dataset):
    def __init__(
        self,
        hparams: argparse.Namespace,
        tokenizer: PreTrainedTokenizerBase,
        format: str,
        split_filename: Callable[[str], str],
        preprocess_and_save: bool = True,
        tokenize_separately: bool = False,
        **load_kwargs,
    ):
        self.split_filename = split_filename
        self.format = format
        assert self.format in {"pkl_dict", "pkl_list", *DATASETS_SUPPORTED_FORMATS}
        self.load_kwargs = load_kwargs
        super().__init__(
            hparams,
            tokenizer,
            preprocess_and_save=preprocess_and_save,
            tokenize_separately=tokenize_separately,
        )

    def load(self, pickle_processor=None) -> DatasetDict:
        split2path = {
            split: os.path.join(self.hparams.data_dir, self.split_filename(split))
            for split in self.all_splits
        }
        if self.format in DATASETS_SUPPORTED_FORMATS:
            dataset_dict = load_dataset(self.format, data_files=split2path, **self.load_kwargs)
            assert isinstance(dataset_dict, DatasetDict)
        elif self.format in ("pkl_dict", "pkl_list"):
            constructor = HFDataset.from_dict if self.format == "pkl_dict" else HFDataset.from_list
            if pickle_processor is None:
                pickle_processor = lambda x: x
            dataset_dict = DatasetDict(
                {
                    split: constructor(pickle_processor(pickle.load(open(path, "rb"))))
                    for split, path in split2path.items()
                }
            )
        else:
            assert False

        return dataset_dict
