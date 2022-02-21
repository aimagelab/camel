from typing import Union, Sequence

import torch

TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]
