import torch


def merge_first_dims(tensor: torch.Tensor, n=2):
    return tensor.reshape(-1, *tensor.shape[n:])


def split_first_dim(tensor: torch.Tensor, dims: list[int]):
    return tensor.reshape(*dims, *tensor.shape[1:])


def lens_to_mask(lens: torch.Tensor, max_len: int = None):
    assert lens.dim() == 1
    if max_len is None:
        max_len = lens.max()
    return torch.arange(max_len, device=lens.device).expand(len(lens), -1) < lens.unsqueeze(1)


def padded_nonzero(tensor: torch.Tensor):
    """
    padded_nonzero(
        [
            [False, True, False],
            [True, True, True],
            [True, False, True],
        ]
    ) = [
        [1, -1, -1],
        [0, 1, 2],
        [0, 2, -1],
    ]
    """
    assert tensor.dim() == 2 and tensor.dtype == torch.bool
    bsz = tensor.shape[0]
    max_per_batch = tensor.sum(-1).max()

    # batch_indices: [0, 0, 0, 1, 2, 3, 3, 5, 6]
    # per_batch_indices: [4, 6, 9, 2, 2, 5, 6, 9, 8]
    batch_indices, per_batch_indices = tensor.nonzero(as_tuple=True)
    # [0, 0, 1, 1, 1, 0, 2, 1]
    last_positions = batch_indices[1:] - batch_indices[:-1]
    assert (last_positions >= 0).all()  # i.e., batch_indices is sorted
    # [1, 1, 1, 2, 3, 4, 4, 6, 7]
    batch_indices_p1 = batch_indices + 1
    # [1, 2, 3, 5, 8, 12, 16, 22, 29]
    batch_indices_cumsum = batch_indices_p1.cumsum(0)
    # [0, 0, 3, 5, 8, 8, 16, 22]
    batch_base = (batch_indices_cumsum[:-1] * (last_positions > 0)).cummax(0).values
    # [1, 2, 3, 2, 3, 4, 8, 6, 7]
    batch_indices_cumsum[1:] -= batch_base
    assert (batch_indices_cumsum % batch_indices_p1 == 0).all()
    # [0, 1, 2, 0, 0, 0, 1, 0, 0]
    batch_arange = batch_indices_cumsum // batch_indices_p1 - 1

    nonzero_indices = per_batch_indices.new_full((bsz, max_per_batch), -1)
    nonzero_indices[batch_indices, batch_arange] = per_batch_indices
    return nonzero_indices
