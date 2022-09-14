import argparse
from typing import Any, Type

from datasets import DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizerBase

from paoding.data.collator import PAD_TYPE
from paoding.data.dataset import Dataset


class ConcatDatasetMeta:
    def __init__(self, dataset_classes: list[Type[Dataset]], dataset_names: list[str] = None):
        self.dataset_classes = dataset_classes
        self.dataset_names = (
            dataset_names
            if dataset_names is not None
            else [str(i) for i in range(len(dataset_classes))]
        )

    def __call__(self, hparams, tokenizer, *args, **kwargs):
        return ConcatDataset(
            hparams,
            tokenizer,
            [
                dataset_cls(hparams, tokenizer, *args, **kwargs)
                for dataset_cls in self.dataset_classes
            ],
            self.dataset_names,
            # *args,
            # **kwargs,
        )

    def add_args(self, parser: argparse.ArgumentParser):
        for dataset_cls in self.dataset_classes:
            dataset_cls.add_args(parser)


class ConcatDataset(Dataset):
    """
    A Dataset object that represents a collection of other Datasets. Currently only supports
    homogenous Datasets with the same output_mode, num_labels, label_key, and metric_names, but if
    necessary in the future, with some changes on the modeling side, I think we could support
    heterogeneous Datasets. The training dataloader implementes **instance-level** (i.e., rather
    than batch-level) mixing of datasets.
    """

    def __init__(
        self,
        hparams: argparse.Namespace,
        tokenizer: PreTrainedTokenizerBase,
        datasets: list[Dataset],
        dataset_names: list[str],
    ):  # , *args, **kwargs):
        assert len(datasets) == len(dataset_names) >= 2
        assert not any("_" in name for name in dataset_names)  # we use _ as a special delimiter
        self.hparams = hparams
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.dataset_dict = self.load()
        # super().__init__(*args, **kwargs, preprocess_and_save=False)

    @property
    def hash_fields(self) -> list[Any]:
        raise NotImplementedError("ConcatDataset doesn't support this property.")
        return [self.__class__.__name__] + [dataset.hash_fields for dataset in self.datasets]

    @property
    def cache_path(self) -> str:
        raise NotImplementedError("ConcatDataset doesn't support this property.")

    @property
    def dev_splits(self) -> list[str]:
        return [
            f"{name}_{split}"
            for dataset, name in zip(self.datasets, self.dataset_names)
            for split in dataset.dev_splits
        ]

    @property
    def test_splits(self) -> list[str]:
        return [
            f"{name}_{split}"
            for dataset, name in zip(self.datasets, self.dataset_names)
            for split in dataset.test_splits
        ]

    @property
    def text_key(self) -> str:
        raise NotImplementedError("ConcatDataset doesn't support this property.")

    @property
    def second_text_key(self) -> str:
        raise NotImplementedError("ConcatDataset doesn't support this property.")

    def _return_common_property(self, name: str) -> Any:
        attr = getattr(self.datasets[0], name)
        assert all(
            getattr(dataset, name) == attr for dataset in self.datasets
        ), f"Children datasets have different {name}: {[getattr(dataset, name) for dataset in self.datasets]}"
        return attr

    @property
    def label_key(self) -> str:
        return self._return_common_property("label_key")

    @property
    def sort_key(self):
        return self._return_common_property("sort_key")

    @property
    def output_mode(self) -> str:
        return self._return_common_property("output_mode")

    @property
    def num_labels(self) -> int:
        return self._return_common_property("num_labels")

    @property
    def metric_names(self) -> list[str]:
        return self._return_common_property("metric_names")

    @property
    def metric_watch_mode(self) -> str:
        return self._return_common_property("metric_watch_mode")

    def pad_token_map(self, split: str) -> dict[str, PAD_TYPE]:
        pad_token_map = self.datasets[0].pad_token_map(split)
        assert all(dataset.pad_token_map(split) == pad_token_map for dataset in self.datasets)
        return pad_token_map

    def assemble_dataset_dicts(self, dataset_dicts: dict[str, DatasetDict]) -> DatasetDict:
        dataset_dict = DatasetDict()
        for name, sub_dataset_dict in dataset_dicts.items():
            for k, v in sub_dataset_dict.items():
                dataset_dict[f"{name}_{k}"] = v
        return dataset_dict

    # def disassemble_dataset_dicts(self, dataset_dict: DatasetDict) -> dict[str, DatasetDict]:
    #     dataset_dicts = {name: {} for name in self.dataset_names}
    #     for k, v in dataset_dict.items():
    #         name, sub_k = k.split("_", maxsplit=1)
    #         dataset_dicts[name][sub_k] = v
    #     return dataset_dicts

    def load(self) -> DatasetDict:
        # return self.assemble_dataset_dicts(
        #     {name: dataset.load() for dataset, name in zip(self.datasets, self.dataset_names)}
        # )
        dataset_dicts = {
            name: dataset.load() for dataset, name in zip(self.datasets, self.dataset_names)
        }
        dataset_dicts = {
            name: dataset.dataset_dict for dataset, name in zip(self.datasets, self.dataset_names)
        }
        dataset_dicts_without_train = {
            name: {k: v for k, v in dataset_dicts[name].items() if k != dataset.train_split}
            for dataset, name in zip(self.datasets, self.dataset_names)
        }
        concated_train = concatenate_datasets(
            [
                dataset_dicts[name][dataset.train_split].remove_columns(
                    set(dataset_dicts[name][dataset.train_split].column_names)
                    - set(dataset.pad_token_map(dataset.train_split))
                    - {dataset.label_key}
                )
                for dataset, name in zip(self.datasets, self.dataset_names)
            ]
        )
        return self.assemble_dataset_dicts(dataset_dicts_without_train) | {
            self.train_split: concated_train
        }

    def preprocess(self, dataset_dict: DatasetDict, map_kwargs: dict = None) -> DatasetDict:
        raise NotImplementedError
        dataset_dicts = self.disassemble_dataset_dicts(dataset_dict)
        dataset_dicts = {
            name: dataset.preprocess(dataset_dicts[name], map_kwargs=map_kwargs)
            for dataset, name in zip(self.datasets, self.dataset_names)
        }
        dataset_dicts_without_train = {
            name: {k: v for k, v in dataset_dicts[name].items() if k != dataset.train_split}
            for dataset, name in zip(self.datasets, self.dataset_names)
        }
        breakpoint()
        concated_train = concatenate_datasets(
            [
                dataset_dicts[name][dataset.train_split]
                for dataset, name in zip(self.datasets, self.dataset_names)
            ]
        )
        return self.assemble_dataset_dicts(dataset_dicts_without_train) | {
            self.train_split: concated_train
        }
