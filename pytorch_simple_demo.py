# flake8: noqa
# fmt: off

import math
from typing import Generic, TypeVar

import numpy as np

import torch

from pyre_extensions import TypeVarTuple
from torch import float32, Tensor

Ts = TypeVarTuple("Ts")
N = TypeVar("N", bound=int)
D = TypeVar("D", bound=int)
DType = TypeVar("DType")

class SomeUnfamiliarModel(Generic[D]):
    def __init__(self, dim_model: D) -> None:
        self.dim_model: D = dim_model

    def forward(self, x: Tensor[float32, N, N]) -> Tensor[float32, N, N, D]:
        seq_len = x.shape[1]

        pos = (
            torch.arange(0, seq_len, dtype=float32, device=x.device)
            .unsqueeze(1)
            .repeat(1, self.dim_model)
        )
        dim = (
            torch.arange(0, self.dim_model, dtype=float32, device=x.device)
            .unsqueeze(0)
            .repeat(seq_len, 1)
        )
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / seq_len))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])

        out1 = x.unsqueeze(-1)
        out2 = pos.unsqueeze(0)
        result = out1 + out2
        return result
