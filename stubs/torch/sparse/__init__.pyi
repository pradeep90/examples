from typing import (
    TypeVar,
    Type,
    overload,
    Optional,
)

from numpy import ndarray

# pyre-ignore[21]: Could not find module `pyre_extensions`. (Spurious error)
from pyre_extensions import (
    TypeVarTuple,
    Unpack,
)
from torch import Tensor
from typing_extensions import Literal as L

from . import nn as nn

DType = TypeVar("DType")
DType2 = TypeVar("DType2")

Ts = TypeVarTuple("Ts")

@overload
def softmax(
    input: Tensor[DType, Unpack[Ts]], dim: int, dtype: Optional[DType2]
) -> Tensor[DType2, Unpack[Ts]]: ...
@overload
def softmax(
    input: Tensor[DType, Unpack[Ts]], dim: int, dtype: Optional[DType] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
