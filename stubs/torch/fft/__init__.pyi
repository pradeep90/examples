from typing import TypeVar, overload

# pyre-ignore[21]: Could not find module `pyre_extensions`. (Spurious error)
from pyre_extensions import (
    TypeVarTuple,
    Unpack,
)
from torch import Tensor, complex64

DType = TypeVar("DType")

Ts = TypeVarTuple("Ts")

@overload
def fft(
    input: Tensor[DType, Unpack[Ts]],
    n: int = ...,
    dim: int = ...,
    norm: str = ...,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[complex64, Unpack[Ts]]: ...
