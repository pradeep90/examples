from typing import TypeVar

import torch
from pyre_extensions import TypeVarTuple, Unpack
from torch import Tensor

Ts = TypeVarTuple("Ts")

def normal_(
    tensor: Tensor[torch.float32, Unpack[Ts]], mean: float = ..., std: float = ...
) -> Tensor[torch.float32, Unpack[Ts]]: ...
