from typing import (
    Any,
    Generic,
    overload,
    Container,
    Iterable,
    Sized,
    SupportsInt,
    SupportsFloat,
    SupportsComplex,
    SupportsBytes,
    SupportsAbs,
    Tuple,
    TypeVar,
)

from typing_extensions import Literal as L
from pyre_extensions import TypeVarTuple, Unpack
from numpy import ndarray
from . import Tensor

Ts = TypeVarTuple("Ts")
T = TypeVar("T")
N1 = TypeVar("N1", bound=int)
N2 = TypeVar("N2", bound=int)
@overload
def argmax(
    tensor: Tensor[T, N1, Unpack[Ts]], axis: L[0] = ...
) -> Tensor[T, Unpack[Ts]]: ...
@overload
def argmax(
    tensor: Tensor[T, N1, N2, Unpack[Ts]], axis: L[1] = ...
) -> Tensor[T, N1, Unpack[Ts]]: ...
