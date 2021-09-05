from collections import defaultdict
from typing import Iterable, Union

import torch
from torch.utils.data._utils.collate import default_collate

T = Union[int, float, bool]


def _find_max_lens(batch: list[dict[str, list[T]]], allow_keys: Iterable[str]) -> int:
    max_lens = defaultdict(int)
    for e in batch:
        for k, v in e.items():
            if k in allow_keys:
                max_lens[k] = max(max_lens[k], len(v))
    return max_lens


def _pad(sequence: list[T], padding_token: T, padding_length: int, padding_side: str) -> list[T]:
    assert padding_side in {"left", "right"}
    if sequence is None:
        return None
    if padding_side == "left":
        return [padding_token] * padding_length + sequence
    else:
        return sequence + [padding_token] * padding_length


def _tensorize(sequence: list[T], name: str, output_mode: str) -> torch.Tensor:
    dtype = torch.long
    if name == "labels" and output_mode == "regression":
        dtype = torch.float
    elif name == "attention_mask":
        dtype = torch.bool
    return torch.tensor(sequence, dtype=dtype)


def collate_fn(
    batch: list[dict[str, list[T]]],
    pad_token_id: int,
    pad_token_type_id: int,
    padding_side: str,
    output_mode: str,
) -> dict[str, torch.Tensor]:
    pad_token_map = {
        "input_ids": pad_token_id,
        "attention_mask": False,
        "token_type_ids": pad_token_type_id,
    }
    max_lens = _find_max_lens(batch, pad_token_map.keys())
    for i, e in enumerate(batch):
        batch[i] = {"labels": e["labels"]} | {
            k: _pad(e[k], pad_token, max_lens[k] - len(e[k]), padding_side)
            for k, pad_token in pad_token_map.items()
        }
        batch[i] = {k: _tensorize(v, k, output_mode) for k, v in batch[i].items()}
    return default_collate(batch)
