from typing import Iterable, TypeAlias

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

PAD_TYPE: TypeAlias = int | float | bool
BATCH_INFO: TypeAlias = dict[str, tuple[torch.dtype, PAD_TYPE]]


def _find_max_shapes(
    batch: list[dict[str, np.ndarray]], allow_keys: Iterable[str]
) -> dict[str, np.ndarray]:
    max_shapes = {}
    for e in batch:
        for k, v in e.items():
            if k not in allow_keys:
                continue
            shape = np.array(v.shape)
            if k not in max_shapes:
                max_shapes[k] = shape
            else:
                try:
                    max_shapes[k] = np.maximum(max_shapes[k], shape)
                except ValueError:  # more informed error message
                    raise ValueError(f"Different shapes for {k}: {max_shapes[k]} vs. {shape}")
    return max_shapes


def _pad(
    sequence: np.ndarray, padding_token: PAD_TYPE, padding_shape: np.ndarray, padding_side: str
) -> np.ndarray:
    assert padding_side in {"left", "right"}
    if sequence is None:
        return None
    padding = [(p, 0) if padding_side == "left" else (0, p) for p in padding_shape]
    return np.pad(sequence, padding, constant_values=padding_token)


def collate_fn(
    batch: list[dict[str, list]], batch_info: BATCH_INFO, padding_side: str
) -> dict[str, torch.Tensor]:
    to_pad_keys = {k for k, (dtype, pad) in batch_info.items() if pad is not None}
    batch = [{k: np.array(v) for k, v in e.items() if k in batch_info} for e in batch]
    max_shapes = _find_max_shapes(batch, to_pad_keys)
    for i, e in enumerate(batch):
        batch[i] = {}
        for k, (dtype, pad_value) in batch_info.items():
            v = e[k]
            if pad_value is not None:
                v = _pad(v, pad_value, max_shapes[k] - np.array(v.shape), padding_side)
            batch[i][k] = torch.tensor(v, dtype=dtype)
    return default_collate(batch)
