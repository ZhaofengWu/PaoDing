"""Adapted from https://github.com/huggingface/transformers/blob/v4.0.0/examples/seq2seq/utils.py"""

import math
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers.file_utils import cached_property


def make_sortish_sampler(
    lens: list[int], batch_size: int, distributed=False, perturb=True
) -> Sampler:
    if distributed:
        return DistributedSortishSampler(lens, batch_size, perturb=perturb)
    else:
        return SortishSampler(lens, batch_size, perturb=perturb)


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness."

    def __init__(self, lens: list[int], batch_size: int, perturb=True):
        self.lens, self.bs, self.perturb = lens, batch_size, perturb

    def __len__(self) -> int:
        return len(self.lens)

    def __iter__(self) -> Iterable:
        return iter(sortish_sampler_indices(self.lens, self.bs, perturb=self.perturb))


def sortish_sampler_indices(lens: list[int], bs: int, perturb=True) -> np.array:
    "Go through the text lens by order of src length with a bit of randomness."
    if not perturb:
        return np.argsort(np.array(lens) * -1).tolist()

    def key_fn(i):
        return lens[i]

    idxs = np.random.permutation(len(lens))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = (
        np.concatenate(np.random.permutation(ck_idx[1:]))
        if len(ck_idx) > 1
        else np.array([], dtype=np.int)
    )
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx.tolist()


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(
        self,
        lens: list[int],
        batch_size: int,
        num_replicas=None,
        rank=None,
        add_extra_examples=True,
        perturb=True,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.lens = lens
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.lens) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.lens)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.perturb = perturb

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(
            sortish_data, self.batch_size, perturb=self.perturb
        )
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.lens)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch
