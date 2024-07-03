from typing import Any, Iterable, TypeAlias

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

PAD_TYPE: TypeAlias = int | float | bool
BATCH_INFO: TypeAlias = dict[str, tuple[torch.dtype, PAD_TYPE]]


def _per_pos_maximum(*lists: list[list[int]]) -> list[int]:
    """
    _per_pos_maximum([3, 2, 1], [1, 2, 3]) -> [3, 2, 3]
    """
    return [max(e) for e in zip(*lists, strict=True)]


def _find_shape_of_ragged_list(l: list[Any]) -> list[int]:
    """Return the shape of a potentially ragged list. For example:
    _find_shape_of_ragged_list([1, 2]) -> [2]
    _find_shape_of_ragged_list([[1, 2, 3], [4]]) -> [2, 3]
    """
    if not isinstance(l, (list, tuple)):
        return []
    return [len(l)] + _per_pos_maximum(*[_find_shape_of_ragged_list(e) for e in l])


def _find_max_shapes(
    batch: list[dict[str, list[Any]]], allow_keys: Iterable[str]
) -> dict[str, np.ndarray]:
    max_shapes = {}
    for e in batch:
        for k, v in e.items():
            if k not in allow_keys:
                continue
            shape = np.array(_find_shape_of_ragged_list(v))
            if k not in max_shapes:
                max_shapes[k] = shape
            else:
                try:
                    max_shapes[k] = np.maximum(max_shapes[k], shape)
                except ValueError:  # more informed error message
                    raise ValueError(f"Different shapes for {k}: {max_shapes[k]} vs. {shape}")
    return max_shapes


def _pad(
    sequence: list[Any], padding_token: PAD_TYPE, max_shape: np.ndarray, padding_side: str
) -> np.ndarray:
    """Pad a potentially nested & ragged list to max_shape."""
    assert padding_side in {"left", "right"}
    if sequence is None:
        return None
    assert isinstance(sequence, list) == (len(max_shape) >= 1)
    if len(max_shape) == 0:
        return np.array(sequence)

    children_shape = max_shape[1:]
    children = np.stack(
        [_pad(x, padding_token, children_shape, padding_side) for x in sequence], axis=0
    )
    curr_padding_len = max_shape[0] - len(children)
    assert curr_padding_len >= 0
    curr_padding = np.full((curr_padding_len,) + tuple(children_shape), padding_token)
    if padding_side == "left":
        return np.concatenate([curr_padding, children])
    else:
        return np.concatenate([children, curr_padding])


def collate_fn(
    batch: list[dict[str, list]], batch_info: BATCH_INFO, padding_side: str
) -> dict[str, torch.Tensor]:
    to_pad_keys = {k for k, (dtype, pad) in batch_info.items() if pad is not None}
    batch = [{k: v for k, v in e.items() if k in batch_info} for e in batch]
    max_shapes = _find_max_shapes(batch, to_pad_keys)
    for i, e in enumerate(batch):
        batch[i] = {}
        for k, (dtype, pad_value) in batch_info.items():
            v = e[k]
            if pad_value is not None:
                v = _pad(v, pad_value, max_shapes[k], padding_side)
            batch[i][k] = torch.tensor(v, dtype=dtype)
    return default_collate(batch)
