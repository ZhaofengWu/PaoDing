from collections import defaultdict

import torch
from torch.utils.data._utils.collate import default_collate


def _find_max_lens(batch, allow_keys):
    max_lens = defaultdict(int)
    for e in batch:
        for k, v in e.items():
            if k in allow_keys:
                max_lens[k] = max(max_lens[k], len(v))
    return max_lens


def _pad(sequence, padding_token, padding_length, padding_side):
    assert padding_side in {"left", "right"}
    if sequence is None:
        return None
    if padding_side == "left":
        return [padding_token] * padding_length + sequence
    else:
        return sequence + [padding_token] * padding_length


def _tensorize(sequence, name, output_mode):
    dtype = torch.long
    if name == "labels" and output_mode == "regression":
        dtype = torch.float
    elif name == "attention_mask":
        dtype = torch.bool
    return torch.tensor(sequence, dtype=dtype)


def collate_fn(batch, pad_token_id, pad_token_type_id, padding_side, output_mode):
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
