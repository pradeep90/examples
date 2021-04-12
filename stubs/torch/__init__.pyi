from pyre_extensions import TypeVarTuple, Unpack, Divide
from typing import Any, Iterable, Generic, TypeVar, Tuple, Type, overload
from typing_extensions import Literal as L
from numpy import ndarray

from .nn import nn as nn

DType = TypeVar("DType")
T = TypeVar("T")
Ts = TypeVarTuple("Ts")
Rs = TypeVarTuple("Rs")
N = TypeVar("N", bound=int)
N1 = TypeVar("N1", bound=int)
N2 = TypeVar("N2", bound=int)

class float32: ...
class float64: ...

double = float64

class Tensor(Generic[DType, Unpack[Ts]]):
    # BEWARE: The type for self must not reuse `Ts`. This is because the type
    # of the object is `Tensor[DType, Unpack[Ts]]`.
    # We are trying to match part of it by using fresh type variables N1 and
    # Rs: `self: Tensor[DType, N1, Unpack[Rs]]`.
    # If we used Ts, then `Ts` would be the one from the object type. We would
    # be saying that the object type `Tensor[DType, Unpack[Ts]]` must match
    # `Tensor[DType, N1, Unpack[Ts]]`, which is absurd.
    @overload
    def size(self: Tensor[DType, N1, Unpack[Rs]], axis: L[0]) -> N1: ...
    @overload
    def size(self: Tensor[DType, N1, N2, Unpack[Rs]], axis: L[1]) -> N2: ...
    @overload
    def split(
        self: Tensor[DType, N1, Unpack[Rs]], split_size_or_sections: N, dim: L[0] = ...
    ) -> Iterable[Tensor[DType, N, Unpack[Rs]]]: ...
    @overload
    def split(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        split_size_or_sections: N,
        dim: L[1] = ...,
    ) -> Iterable[Tensor[DType, N1, N, Unpack[Rs]]]: ...

    def item(self: Tensor[DType]) -> DType: ...
    def backward(self) -> None: ...
    def __getitem__(self, item: Any) -> Any: ...
    def detach(self: T) -> T: ...
    def numpy(self) -> ndarray[DType, Unpack[Ts]]: ...


@overload
def zeros(*size: Unpack[Ts], dtype: Type[DType]) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def zeros(
    *size: Unpack[Ts], dtype: Type[float32] = ...
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def zeros(
    *size: Tuple[Unpack[Ts]], dtype: Type[DType]
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def zeros(
    *size: Tuple[Unpack[Ts]], dtype: Type[float32] = ...
) -> Tensor[float32, Unpack[Ts]]: ...

# This takes a list of tensors and concatenates them across an axis. We don't
# know the length of the list and thus can't tell the final dimensions of the
# tensor.
@overload
def cat(
    tensors: Iterable[Tensor[DType, Unpack[Ts]]], dim: int, *, out: Any = ...
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...

save: Any
manual_seed: Any
load: Any
from_numpy: Any
no_grad: Any
