import torch
from torch import Tensor


def one_hot_to_index(one_hot: Tensor) -> Tensor:
    """
    Converts a one-hot tensor into a tensor with corresponding indexes
    """
    device, dtype = one_hot.device, one_hot.dtype
    vocab_size = one_hot.shape[-1]
    oh2idx = torch.tensor(range(vocab_size), dtype=dtype, device=device)
    return (one_hot @ oh2idx.unsqueeze(dim=1)).long().squeeze(dim=-1)
