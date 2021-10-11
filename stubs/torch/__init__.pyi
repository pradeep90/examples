import builtins
from typing import (
    Any,
    Iterable,
    Generic,
    List,
    TypeVar,
    Tuple,
    Type,
    overload,
    Optional,
    Sequence,
    Union,
)

from numpy import ndarray

# pyre-ignore[21]: Could not find module `pyre_extensions`. (Spurious error)
from pyre_extensions import (
    TypeVarTuple,
    Unpack,
    Divide,
    Add,
    Multiply,
    Broadcast,
    Product,
)
from typing_extensions import Literal as L

from . import nn as nn
from . import sparse as sparse

DType = TypeVar("DType")
DType2 = TypeVar("DType2")
Layout = TypeVar("Layout")
Device = TypeVar("Device")
Wild = TypeVar("Wild")

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
Rs = TypeVarTuple("Rs")
Rs2 = TypeVarTuple("Rs2")
Qs = TypeVarTuple("Qs")
N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
B = TypeVar("B", bound=int)
P = TypeVar("P", bound=int)
R = TypeVar("R", bound=int)
N1 = TypeVar("N1", bound=int)
N2 = TypeVar("N2", bound=int)
N3 = TypeVar("N3", bound=int)
N4 = TypeVar("N4", bound=int)
N5 = TypeVar("N5", bound=int)
N6 = TypeVar("N6", bound=int)

builtin_bool = builtins.bool
builtin_float = builtins.float

# These are torch's datatypes, which have the same names as the builtins.
class complex64: ...
class complex128: ...
class float32: ...
class float64: ...
class int64: ...
class int32: ...
class bool: ...
class memory_format: ...

float = float32
double = float64

class long: ...
class layout: ...

class device:
    def __init__(self, device_str: str): ...

class memory_format: ...

class Tensor(Generic[DType, Unpack[Ts]]):
    @property
    def device(self) -> device: ...
    @property
    def dtype(self) -> Type[DType]: ...
    def long(self) -> "LongTensor[DType, Unpack[Ts]]": ...
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
    def size(self: Tensor[DType, Unpack[Rs], N1], axis: L[-1]) -> N1: ...
    @overload
    def size(self: Tensor[DType, Unpack[Rs], N1, N2], axis: L[-2]) -> N1: ...
    @overload
    def size(self: Tensor[DType, Unpack[Rs]]) -> Tuple[Unpack[Rs]]: ...
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
    def numel(self) -> int: ...
    def backward(self) -> None: ...
    @overload
    def __getitem__(
        self: Tensor[DType, N, Unpack[Rs]], item: L[0]
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def __getitem__(
        self: Tensor[DType, Unpack[Rs]], item: None
    ) -> Tensor[DType, L[1], Unpack[Rs]]: ...
    @overload
    def __getitem__(
        self: Tensor[DType, Unpack[Rs]], item: Tensor[bool, Unpack[Rs]]
    ) -> Tensor[DType, int]: ...
    @overload
    def __getitem__(self, item: Any) -> Any: ...
    @overload
    def expand(
        self: Tensor[DType, Unpack[Rs]], shape: Tuple[Unpack[Rs2]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    def detach(self: T) -> T: ...
    def numpy(self) -> ndarray[DType, Unpack[Ts]]: ...
    shape: Tuple[Unpack[Ts]]
    ndim: int
    @overload
    def to(
        self: Tensor[DType, Unpack[Rs]], dtype: Type[T], device: Device = ...
    ) -> Tensor[T, Unpack[Rs]]: ...
    @overload
    def to(
        self: Tensor[DType, Unpack[Rs]], device: device
    ) -> Tensor[DType, Unpack[Rs]]: ...
    device: device
    @overload
    def __add__(
        self: Tensor[DType, Unpack[Rs]], other: Tensor[DType, Unpack[Rs2]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __add__(
        self: Tensor[DType, Unpack[Rs]],
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def __iadd__(
        self, other: Tensor[DType, Unpack[Rs]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __iadd__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __radd__(
        self: Tensor[DType, Unpack[Rs]], other: Tensor[DType, Unpack[Rs2]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __radd__(
        self: Tensor[DType, Unpack[Rs]],
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def __sub__(
        self: Tensor[DType, Unpack[Rs]], other: Tensor[DType, Unpack[Rs2]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __sub__(
        self: Tensor[DType, Unpack[Rs]],
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def __isub__(
        self, other: Tensor[DType, Unpack[Rs]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __isub__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def __rsub__(
        self: Tensor[DType, Unpack[Rs]], other: Tensor[DType, Unpack[Rs2]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __rsub__(
        self: Tensor[DType, Unpack[Rs]],
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def __mul__(
        self: Tensor[DType, Unpack[Rs]],
        other: Tensor[DType, Unpack[Rs2]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __mul__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __imul__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __imul__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __rmul__(
        self: Tensor[DType, Unpack[Rs]],
        other: Tensor[DType, Unpack[Rs2]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __rmul__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __truediv__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __truediv__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[float32, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __itruediv__(
        self,
        other: builtin_float,
    ) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def __itruediv__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __rtruediv__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __floordiv__(
        self,
        other: int,
    ) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def __floordiv__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __ifloordiv__(
        self,
        other: int,
    ) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def __ifloordiv__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __rfloordiv__(
        self,
        other: int,
    ) -> Tensor[DType, Unpack[Ts]]: ...
    def __invert__(self) -> Tensor[DType, Unpack[Ts]]: ...
    def __neg__(self) -> Tensor[DType, Unpack[Ts]]: ...
    def __iand__(
        self: Tensor[bool, Unpack[Rs]],
        other: Tensor[bool, Unpack[Rs2]],
    ) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    def __and__(
        self: Tensor[bool, Unpack[Rs]],
        other: Tensor[bool, Unpack[Rs2]],
    ) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __matmul__(
        self: Tensor[DType, N1],
        other: Tensor[DType, N1],
    ) -> Tensor[DType]: ...
    @overload
    def __matmul__(
        self: Tensor[DType, Unpack[Rs], N1, N2],
        other: Tensor[DType, Unpack[Qs], N2, N3],
    ) -> Tensor[
        DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Qs]]]], N1, N3
    ]: ...
    def __ne__(
        self: Tensor[DType, Unpack[Rs]], other: DType
    ) -> Tensor[bool, Unpack[Rs]]: ...
    @overload
    def all(
        self: Tensor[DType, Unpack[Ts]],
    ) -> Tensor[bool, L[1]]: ...
    @overload
    def all(
        self: Tensor[DType, N, Unpack[Ts]],
        dim: L[0],
    ) -> Tensor[bool, Unpack[Ts]]: ...
    @overload
    def all(
        self: Tensor[DType, N1, N2, Unpack[Ts]],
        dim: L[1],
    ) -> Tensor[bool, N1, Unpack[Ts]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[True],
    ) -> Tensor[int64, L[1], Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[False] = ...,
    ) -> Tensor[int64, Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[True],
    ) -> Tensor[int64, N1, L[1], Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[False] = ...,
    ) -> Tensor[int64, N1, Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[True],
    ) -> Tensor[int64, N1, N2, L[1], Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[False] = ...,
    ) -> Tensor[int64, N1, N2, Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[True],
    ) -> Tensor[int64, Unpack[Rs], L[1]]: ...
    @overload
    def argmin(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[False] = ...,
    ) -> Tensor[int64, Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, Unpack[Rs]],
        dim: L[None] = ...,
        keepdim: builtins.bool = ...,
    ) -> Tensor[int64]: ...
    def clone(
        input, *, memory_format: Optional[memory_format] = ...
    ) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def count_nonzero(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
    ) -> Tensor[int64, Unpack[Rs]]: ...
    @overload
    def count_nonzero(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
    ) -> Tensor[int64, N1, Unpack[Rs]]: ...
    @overload
    def count_nonzero(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
    ) -> Tensor[int64, N1, N2, Unpack[Rs]]: ...
    @overload
    def count_nonzero(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
    ) -> Tensor[int64, Unpack[Rs]]: ...
    @overload
    def count_nonzero(
        self: Tensor[DType, Unpack[Rs]],
        dim: L[None] = ...,
        keepdim: builtins.bool = ...,
    ) -> Tensor[int64]: ...
    def indices(self) -> Tensor: ...
    @overload
    def max(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[True],
    ) -> Tensor[DType, L[1], Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[True],
    ) -> Tensor[DType, N1, L[1], Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, N1, Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[True],
    ) -> Tensor[DType, N1, N2, L[1], Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[True],
    ) -> Tensor[DType, Unpack[Rs], L[1]]: ...
    @overload
    def max(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, Unpack[Rs], N1, N2],
        dim: L[-2],
        keepdim: L[True],
    ) -> Tensor[DType, Unpack[Rs], L[1], N2]: ...
    @overload
    def max(
        self: Tensor[DType, Unpack[Rs], N1, N2],
        dim: L[-2],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, Unpack[Rs], N2]: ...
    @overload
    def max(
        self: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[DType]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[True],
    ) -> Tensor[DType, L[1], Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[True],
    ) -> Tensor[DType, N1, L[1], Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, N1, Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[True],
    ) -> Tensor[DType, N1, N2, L[1], Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[True],
    ) -> Tensor[DType, Unpack[Rs], L[1]]: ...
    @overload
    def mean(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, Unpack[Rs]],
        dim: L[None] = ...,
        keepdim: builtins.bool = ...,
    ) -> Tensor[DType]: ...
    def bitwise_not(self) -> Tensor[DType, Unpack[Ts]]: ...
    def bitwise_not_(self) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def diff(
        self: Tensor[DType, Unpack[Rs], Add[N1, L[1]], N2],
        dim: L[-2],
    ) -> Tensor[DType, Unpack[Rs], N1, N2]: ...
    @overload
    def diff(
        self: Tensor[DType, Add[N, L[1]], Unpack[Rs]],
        dim: L[0],
    ) -> Tensor[DType, N, Unpack[Rs]]: ...
    @overload
    def diff(
        self: Tensor[DType, N1, Add[N2, L[1]], Unpack[Rs]],
        dim: L[1],
    ) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
    @overload
    def diff(
        self: Tensor[DType, Unpack[Rs], Add[N, L[1]]],
        dim: L[-1] = ...,
    ) -> Tensor[DType, Unpack[Rs], N]: ...
    def is_sparse(self) -> builtins.bool: ...
    def coalesce(self: Tensor[DType, Unpack[Rs]]) -> Tensor[DType, Unpack[Rs]]: ...
    def values(self: Tensor[DType, Unpack[Rs]]) -> Tensor[DType, Unpack[Rs]]: ...
    def to_sparse(self: Tensor[DType, Unpack[Ts]]) -> Tensor[DType, Unpack[Ts]]: ...
    def float(self: Tensor[DType, Unpack[Rs]]) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def item(self: Tensor[DType, L[1]]) -> DType: ...
    @overload
    def __eq__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __eq__(
        self,
        other: object,
    ) -> Tensor[bool, Unpack[Ts]]: ...
    def argsort(
        self, dim: int = ..., descending: builtin_bool = ...
    ) -> Tensor[DType, Unpack[Ts]]: ...
    def bmm(
        self: Tensor[DType, B, N, M], mat2: Tensor[DType, B, M, P]
    ) -> Tensor[DType, B, N, P]: ...
    def diag_embed(
        self: Tensor[DType, Unpack[Rs], N]
    ) -> Tensor[DType, Unpack[Rs], N, N]: ...
    @overload
    def matmul(
        self: Tensor[DType, N1],
        other: Tensor[DType, N1],
    ) -> Tensor[DType]: ...
    @overload
    def matmul(
        self: Tensor[DType, Unpack[Rs], N1, N2],
        other: Tensor[DType, Unpack[Qs], N2, N3],
    ) -> Tensor[
        DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Qs]]]], N1, N3
    ]: ...
    def multinomial(
        self: Tensor[DType, Unpack[Rs], N1],
        num_samples: N2,
        replacement: builtins.bool = ...,
        *,
        generator: Optional[Generator] = ...,
    ) -> Tensor[DType, Unpack[Rs], N2]: ...
    @overload
    def unsqueeze(
        self: Tensor[DType, Unpack[Rs]], dim: L[-1]
    ) -> Tensor[DType, Unpack[Rs], L[1]]: ...
    @overload
    def unsqueeze(
        self: Tensor[DType, Unpack[Rs]], dim: L[0]
    ) -> Tensor[DType, L[1], Unpack[Rs]]: ...
    @overload
    def unsqueeze(
        self: Tensor[DType, N, Unpack[Rs]], dim: L[1]
    ) -> Tensor[DType, N, L[1], Unpack[Rs]]: ...
    @overload
    def unsqueeze(
        self: Tensor[DType, N1, N2, Unpack[Rs]], dim: L[2]
    ) -> Tensor[DType, N1, N2, L[1], Unpack[Rs]]: ...
    @overload
    def repeat(
        self: Tensor[DType, N1], size1: N2
    ) -> Tensor[DType, Multiply[N1, N2]]: ...
    @overload
    def repeat(
        self: Tensor[DType, N1, N2], size1: N3, size2: N4
    ) -> Tensor[DType, Multiply[N1, N3], Multiply[N2, N4]]: ...
    @overload
    def repeat(
        self: Tensor[DType, N1, N2, N3], size1: N4, size2: N5, size3: N6
    ) -> Tensor[DType, Multiply[N1, N4], Multiply[N2, N5], Multiply[N3, N6]]: ...
    @overload
    def repeat_interleave(
        self: Tensor[DType, Unpack[Rs], N1], repeats: N, dim: L[-1]
    ) -> Tensor[DType, Unpack[Rs], Multiply[N1, N]]: ...
    @overload
    def repeat_interleave(
        self: Tensor[DType, N1, Unpack[Rs]], repeats: N, dim: L[0]
    ) -> Tensor[DType, Multiply[N1, N], Unpack[Rs]]: ...
    @overload
    def repeat_interleave(
        self: Tensor[DType, N1, N2, Unpack[Rs]], repeats: N, dim: L[1]
    ) -> Tensor[DType, N1, Multiply[N2, N], Unpack[Rs]]: ...
    @overload
    def repeat_interleave(
        self: Tensor[DType, Unpack[Rs]], repeats: N, dim: L[None] = ...
    ) -> Tensor[DType, Product[N, Unpack[Rs]]]: ...
    # The output shape here depends on the contents of `repeats`, so give up.
    @overload
    def repeat_interleave(
        input: Tensor[DType, Unpack[Rs]], repeats: Tensor, dim: int = ...
    ) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...
    def __setitem__(self, item: object, other: object) -> None: ...
    @overload
    def softmax(self, dim: int) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def softmax(self, dim: int, dtype: Type[DType2]) -> Tensor[DType2, Unpack[Ts]]: ...
    def type_as(
        self, other: Tensor[DType2, Unpack[Rs]]
    ) -> Tensor[DType2, Unpack[Rs]]: ...
    @overload
    def view(
        self: Tensor[DType, Unpack[Rs]], *shape: Unpack[Tuple[L[-1], Unpack[Rs2]]]
    ) -> Tensor[
        DType, Divide[Product[Unpack[Rs]], Product[Unpack[Rs2]]], Unpack[Rs2]
    ]: ...
    @overload
    def view(
        self: Tensor[DType, Unpack[Rs]], *shape: Unpack[Tuple[N1, L[-1], Unpack[Rs2]]]
    ) -> Tensor[
        DType, N1, Divide[Product[Unpack[Rs]], Product[N1, Unpack[Rs2]]], Unpack[Rs2]
    ]: ...
    @overload
    def view(
        self: Tensor[DType, Unpack[Rs]], *shape: Unpack[Tuple[Unpack[Rs2], L[-1]]]
    ) -> Tensor[
        DType, Unpack[Rs2], Divide[Product[Unpack[Rs]], Product[Unpack[Rs2]]]
    ]: ...
    @overload
    def view(self, *shape: Unpack[Rs]) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def transpose(
        self: Tensor[DType, Unpack[Rs], N1, N2], dim0: L[-2], dim1: L[-1]
    ) -> Tensor[DType, Unpack[Rs], N2, N1]: ...
    @overload
    def transpose(
        self: Tensor[DType, Unpack[Rs], N1, N2], dim0: L[-1], dim1: L[-2]
    ) -> Tensor[DType, Unpack[Rs], N2, N1]: ...
    @overload
    def transpose(
        self: Tensor[DType, N1, N2, Unpack[Rs]], dim0: L[0], dim1: L[1]
    ) -> Tensor[DType, N2, N1, Unpack[Rs]]: ...
    @overload
    def transpose(
        self: Tensor[DType, N1, N2, Unpack[Rs]], dim0: L[1], dim1: L[0]
    ) -> Tensor[DType, N2, N1, Unpack[Rs]]: ...
    @overload
    def transpose(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]], dim0: L[1], dim1: L[2]
    ) -> Tensor[DType, N1, N3, N2, Unpack[Rs]]: ...
    @overload
    def flatten(
        self: Tensor[DType, N1, Unpack[Rs], N2],
        start_dim: L[0] = ...,
        end_dim: L[-1] = ...,
    ) -> Tensor[DType, Product[N1, Unpack[Rs], N2]]: ...
    @overload
    def flatten(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        start_dim: L[0] = ...,
        end_dim: L[1] = ...,
    ) -> Tensor[DType, Multiply[N1, N2], Unpack[Rs]]: ...
    @overload
    def flatten(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        start_dim: L[1] = ...,
        end_dim: L[2] = ...,
    ) -> Tensor[DType, N1, Multiply[N2, N3], Unpack[Rs]]: ...
    @overload
    def flatten(
        self: Tensor[DType, N1, N2, N3, N4, Unpack[Rs]],
        start_dim: L[2] = ...,
        end_dim: L[3] = ...,
    ) -> Tensor[DType, N1, N2, Multiply[N3, N4], Unpack[Rs]]: ...
    @overload
    def flatten(
        self: Tensor[DType],
        start_dim: L[0] = ...,
        end_dim: L[0] = ...,
    ) -> Tensor[DType, L[1]]: ...
    @overload
    def __gt__(
        self: Tensor[DType, Unpack[Rs]], x: DType
    ) -> Tensor[bool, Unpack[Rs]]: ...
    @overload
    def __gt__(
        self: Tensor[float32, Unpack[Rs]], x: builtin_float
    ) -> Tensor[bool, Unpack[Rs]]: ...
    def logical_and(
        self,
        other: Tensor[DType2, Unpack[Rs]],
        *,
        out: Optional[Tensor] = ...,
    ) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    def logical_and_(
        self,
        other: Tensor[DType2, Unpack[Rs]],
        *,
        out: Optional[Tensor] = ...,
    ) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def reshape(
        self: Tensor[DType, Unpack[Rs]], *shape: Unpack[Tuple[L[-1], Unpack[Rs2]]]
    ) -> Tensor[
        DType, Divide[Product[Unpack[Rs]], Product[Unpack[Rs2]]], Unpack[Rs2]
    ]: ...
    @overload
    def reshape(
        self: Tensor[DType, Unpack[Rs]], *shape: Unpack[Tuple[N1, L[-1], Unpack[Rs2]]]
    ) -> Tensor[
        DType, N1, Divide[Product[Unpack[Rs]], Product[N1, Unpack[Rs2]]], Unpack[Rs2]
    ]: ...
    @overload
    def reshape(self, *shape: Unpack[Rs]) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def unbind(
        self: Tensor[DType, Unpack[Rs], N], dim: L[-1]
    ) -> Tuple[Tensor[DType, Unpack[Rs]], ...]: ...
    @overload
    def unbind(
        self: Tensor[DType, N, N1, Unpack[Rs]], dim: L[1]
    ) -> Tuple[Tensor[DType, N, Unpack[Rs]], ...]: ...
    @overload
    def unbind(
        self: Tensor[DType, N, Unpack[Rs]], dim: L[0] = ...
    ) -> Tuple[Tensor[DType, Unpack[Rs]], ...]: ...
    def sign(self, *, out: Optional[Tensor] = ...) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def sum(
        self: Tensor[DType, N1, Unpack[Rs]], dim: L[0], *, dtype: Optional[device] = ...
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def sum(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        *,
        dtype: Optional[device] = ...,
    ) -> Tensor[DType, N1, Unpack[Rs]]: ...
    @overload
    def sum(
        self: Tensor[DType, Unpack[Rs], N], dim: L[-1], *, dtype: Optional[device] = ...
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def sum(
        self: Tensor[DType, Unpack[Rs], N1, N2],
        dim: L[-2],
        *,
        dtype: Optional[device] = ...,
    ) -> Tensor[DType, Unpack[Rs], N2]: ...
    @overload
    def sum(
        self: Tensor[DType, Unpack[Rs]],
        dim: L[None] = ...,
        *,
        dtype: Optional[device] = ...,
    ) -> Tensor[DType]: ...
    def cumsum(
        self: Tensor[DType, Unpack[Rs]], dim: int = ..., dtype: Optional[device] = ...
    ) -> Tensor[DType, Unpack[Rs]]: ...
    def contiguous(input: Tensor[DType, Unpack[Rs]]) -> Tensor[DType, Unpack[Rs]]: ...

class LongTensor(Tensor[DType, Unpack[Ts]], Generic[DType, Unpack[Ts]]):
    @overload
    def __getitem__(
        self: LongTensor[DType, Unpack[Rs], N], val: Tuple[object, None]
    ) -> LongTensor[DType, Unpack[Tuple[Any, ...]]]: ...
    @overload
    def __getitem__(
        self: LongTensor[DType, Unpack[Rs], N], val: Tuple[None, object]
    ) -> LongTensor[DType, Unpack[Tuple[Any, ...]]]: ...
    @overload
    def __getitem__(
        self: LongTensor[DType, Unpack[Rs], N], val: slice
    ) -> LongTensor[DType, Unpack[Tuple[Any, ...]]]: ...
    def __eq__(
        self: LongTensor[DType, Unpack[Rs]],
        other: LongTensor[DType, Unpack[Rs]],
    ) -> LongTensor[bool, Unpack[Rs]]: ...

# NOTE: These `torch` functions below have a method counterpart in `Tensor`. So,
# if you update the stubs here, please update the method stub as well.

def allclose(
    input: Tensor,
    other: Tensor,
    rtol: builtin_float = ...,
    atol: builtin_float = ...,
    equal_nan: builtins.bool = ...,
) -> builtins.bool: ...
def bitwise_not(
    input: Tensor[DType, Unpack[Ts]], *, out: Optional[Tensor] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
def einsum(
    equation: str,
    *operands: Tensor,
) -> Tensor: ...
@overload
def eye(
    n: N,
    *,
    dtype: Type[float32] = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Union[device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[float32, N, N]: ...
@overload
def eye(
    n: N,
    m: M,
    *,
    dtype: Type[float32] = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Union[device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[float32, N, M]: ...
@overload
def eye(
    n: N,
    *,
    dtype: DType,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Union[device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, N, N]: ...
@overload
def eye(
    n: N,
    m: M,
    *,
    dtype: DType,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Union[device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, N, M]: ...
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
@overload
def ones(*size: Unpack[Ts]) -> Tensor[float, Unpack[Ts]]: ...
@overload
def ones(
    *size: Unpack[Ts], dtype: Type[DType] = ..., device: device = ...
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def ones_like(
    input: Tensor[DType, Unpack[Ts]],
    *,
    dtype: Type[DType2],
    memory_format: Optional[memory_format] = ...,
    layout: Optional[layout] = ...,
    device: Union[device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType2, Unpack[Ts]]: ...
@overload
def ones_like(
    input: Tensor[DType, Unpack[Ts]],
    *,
    memory_format: Optional[memory_format] = ...,
    dtype: Type[DType] = ...,
    layout: Optional[layout] = ...,
    device: Union[device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
def rand(*size: Unpack[Ts]) -> Tensor[float, Unpack[Ts]]: ...
def tril(
    x: Tensor[DType, Unpack[Ts]], diagonal: int = ...
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def arange(
    end: N1,
    *,
    out: Optional[int] = ...,
    dtype: Type[int64] = ...,
    layout: Type[layout] = ...,
    device: Type[device] = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[int64, N1]: ...
@overload
def arange(
    start: N1,
    end: N2,
    *,
    out: Optional[int] = ...,
    dtype: Type[int64] = ...,
    layout: Type[layout] = ...,
    device: Type[device] = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[int64, Add[N2, Multiply[L[-1], N1]]]: ...
@overload
def arange(
    start: N1,
    end: N2,
    step: N3,
    out: Optional[int] = ...,
    dtype: Type[int64] = ...,
    layout: Type[layout] = ...,
    device: Type[device] = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[int64, Divide[Add[N2, Multiply[L[-1], N1]], N3]]: ...

# dtype is explicitly provided.
@overload
def arange(
    end: N1,
    *,
    dtype: Type[DType],
    out: Optional[int] = ...,
    layout: Type[layout] = ...,
    device: Type[device] = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, N1]: ...
@overload
def arange(
    start: N1,
    end: N2,
    *,
    dtype: Type[DType],
    out: Optional[int] = ...,
    layout: Type[layout] = ...,
    device: Type[device] = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, Add[N2, Multiply[L[-1], N1]]]: ...
@overload
def arange(
    start: N1,
    end: N2,
    step: N3,
    dtype: Type[DType],
    out: Optional[int] = ...,
    dtype: Type[int64] = ...,
    layout: Type[layout] = ...,
    device: Type[Device] = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, Divide[Add[N2, Multiply[L[-1], N1]], N]]: ...
@overload
def arange(
    end: builtin_float,
    start: builtin_float = ...,
    step: builtin_float = ...,
    out: Optional[int] = ...,
    dtype: Type[DType] = ...,
    layout: Type[Parameter] = ...,
    device: device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, int]: ...
@overload
def argmin(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[True],
) -> Tensor[int64, L[1], Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[False] = ...,
) -> Tensor[int64, Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[True],
) -> Tensor[int64, N1, L[1], Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[False] = ...,
) -> Tensor[int64, N1, Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[True],
) -> Tensor[int64, N1, N2, L[1], Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[False] = ...,
) -> Tensor[int64, N1, N2, Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[True],
) -> Tensor[int64, Unpack[Rs], L[1]]: ...
@overload
def argmin(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[False] = ...,
) -> Tensor[int64, Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, Unpack[Rs]],
    dim: L[None] = ...,
    keepdim: builtins.bool = ...,
) -> Tensor[int64]: ...
def bmm(
    input: Tensor[DType, B, N, M], mat2: Tensor[DType, B, M, P]
) -> Tensor[DType, B, N, P]: ...
def diag(
    input: Tensor[DType, Unpack[Tuple[Any, ...]]],
    diagonal: int = ...,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...
def diagonal(
    input: Tensor[DType, Unpack[Tuple[Any, ...]]],
    offset: int = ...,
    dim1: int = ...,
    dim2: int = ...,
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...
def diag_embed(
    input: Tensor[DType, Unpack[Tuple[Any, ...]]],
    offset: int = ...,
    dim1: int = ...,
    dim2: int = ...,
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...
def logical_and(
    input: Tensor[DType, Unpack[Ts]],
    other: Tensor[DType2, Unpack[Rs]],
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
@overload
def max(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[True],
) -> Tensor[DType, L[1], Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[False] = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[True],
) -> Tensor[DType, N1, L[1], Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[False] = ...,
) -> Tensor[DType, N1, Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[True],
) -> Tensor[DType, N1, N2, L[1], Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[False] = ...,
) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[True],
) -> Tensor[DType, Unpack[Rs], L[1]]: ...
@overload
def max(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[False] = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, Unpack[Rs], N1, N2],
    dim: L[-2],
    keepdim: L[True],
) -> Tensor[DType, Unpack[Rs], L[1], N2]: ...
@overload
def max(
    input: Tensor[DType, Unpack[Rs], N1, N2],
    dim: L[-2],
    keepdim: L[False] = ...,
) -> Tensor[DType, Unpack[Rs], N2]: ...
@overload
def max(
    input: Tensor[DType, Unpack[Rs]],
) -> Tensor[DType]: ...
@overload
def mean(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[True],
) -> Tensor[DType, L[1], Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[False] = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[True],
) -> Tensor[DType, N1, L[1], Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[False] = ...,
) -> Tensor[DType, N1, Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[True],
) -> Tensor[DType, N1, N2, L[1], Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[False] = ...,
) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[True],
) -> Tensor[DType, Unpack[Rs], L[1]]: ...
@overload
def mean(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[False] = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, Unpack[Rs]],
    dim: L[None] = ...,
    keepdim: builtins.bool = ...,
) -> Tensor[DType]: ...
@overload
def meshgrid(tensor1: Tensor[DType, N1]) -> Tuple[Tensor[DType, N1]]: ...
@overload
def meshgrid(
    tensor1: Tensor[DType, N1],
    tensor2: Tensor[DType, N2],
) -> Tuple[Tensor[DType, N1, N2], Tensor[DType, N1, N2]]: ...
@overload
def meshgrid(
    tensor1: Tensor[DType, N1],
    tensor2: Tensor[DType, N2],
    tensor3: Tensor[DType, N3],
) -> Tuple[
    Tensor[DType, N1, N2, N3], Tensor[DType, N1, N2, N3], Tensor[DType, N1, N2, N3]
]: ...
@overload
def meshgrid(*tensors: Tensor) -> Tuple[Tensor, ...]: ...
def rand_like(
    input: Tensor[Wild, Unpack[Ts]], dtype: Type[DType]
) -> Tensor[DType, Unpack[Ts]]: ...
def nonzero(
    input: Tensor[DType, Unpack[Ts]], as_tuple: L[False] = ...
) -> LongTensor[DType, int, int]: ...
@overload
def repeat_interleave(
    input: Tensor[DType, Unpack[Rs], N1], repeats: N, dim: L[-1]
) -> Tensor[DType, Unpack[Rs], Multiply[N1, N]]: ...
@overload
def repeat_interleave(
    input: Tensor[DType, N1, Unpack[Rs]], repeats: N, dim: L[0]
) -> Tensor[DType, Multiply[N1, N], Unpack[Rs]]: ...
@overload
def repeat_interleave(
    input: Tensor[DType, N1, N2, Unpack[Rs]], repeats: N, dim: L[1]
) -> Tensor[DType, N1, Multiply[N2, N], Unpack[Rs]]: ...
@overload
def repeat_interleave(
    input: Tensor[DType, Unpack[Rs]], repeats: N, dim: L[None] = ...
) -> Tensor[DType, Product[N, Unpack[Rs]]]: ...

# The output shape here depends on the contents of `repeats`, so give up.
@overload
def repeat_interleave(
    input: Tensor[DType, Unpack[Rs]], repeats: Tensor, dim: int = ...
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...
@overload
def stack(
    tensors: Tuple[Tensor[DType, N, Unpack[Ts]], Tensor[DType, N, Unpack[Ts]]],
    dim: L[1],
    *,
    out: Optional[Tensor[DType, L[2], Unpack[Ts]]] = ...,
) -> Tensor[DType, N, L[2], Unpack[Ts]]: ...
@overload
def stack(
    tensors: Tuple[Tensor[DType, Unpack[Ts]], Tensor[DType, Unpack[Ts]]],
    dim: L[0] = ...,
    *,
    out: Optional[Tensor[DType, L[2], Unpack[Ts]]] = ...,
) -> Tensor[DType, L[2], Unpack[Ts]]: ...
@overload
def stack(
    tensors: Tuple[
        Tensor[DType, N, Unpack[Ts]],
        Tensor[DType, N, Unpack[Ts]],
        Tensor[DType, N, Unpack[Ts]],
    ],
    dim: L[1],
    *,
    out: Optional[Tensor[DType, L[3], Unpack[Ts]]] = ...,
) -> Tensor[DType, N, L[3], Unpack[Ts]]: ...
@overload
def stack(
    tensors: Tuple[
        Tensor[DType, Unpack[Ts]],
        Tensor[DType, Unpack[Ts]],
        Tensor[DType, Unpack[Ts]],
    ],
    dim: L[0] = ...,
    *,
    out: Optional[Tensor[DType, L[3], Unpack[Ts]]] = ...,
) -> Tensor[DType, L[3], Unpack[Ts]]: ...
@overload
def stack(
    tensors: Tuple[Any, ...],
    dim: N = ...,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor: ...
def cdist(
    input: Tensor[DType, Unpack[Ts], P, M],
    other: Tensor[DType, Unpack[Rs], R, M],
    p: builtin_float = ...,
    compute_mode: str = ...,
) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]], P, R]: ...
def clone(
    input: Tensor[DType, Unpack[Ts]], *, memory_format: Optional[memory_format] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def count_nonzero(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
) -> Tensor[int64, Unpack[Rs]]: ...
@overload
def count_nonzero(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
) -> Tensor[int64, N1, Unpack[Rs]]: ...
@overload
def count_nonzero(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
) -> Tensor[int64, N1, N2, Unpack[Rs]]: ...
@overload
def count_nonzero(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
) -> Tensor[int64, Unpack[Rs]]: ...
@overload
def count_nonzero(
    input: Tensor[DType, Unpack[Rs]],
    dim: L[None] = ...,
    keepdim: builtins.bool = ...,
) -> Tensor[int64]: ...
@overload
def sum(
    input: Tensor[DType, N1, Unpack[Rs]], dim: L[0], *, dtype: Optional[device] = ...
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def sum(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    *,
    dtype: Optional[device] = ...,
) -> Tensor[DType, N1, Unpack[Rs]]: ...
@overload
def sum(
    input: Tensor[DType, Unpack[Rs], N], dim: L[-1], *, dtype: Optional[device] = ...
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def sum(
    input: Tensor[DType, Unpack[Rs], N1, N2],
    dim: L[-2],
    *,
    dtype: Optional[device] = ...,
) -> Tensor[DType, Unpack[Rs], N2]: ...
@overload
def sum(
    input: Tensor[DType, Unpack[Rs]],
    dim: L[None] = ...,
    *,
    dtype: Optional[device] = ...,
) -> Tensor[DType]: ...
@overload
def sin(
    input: Tensor[DType, Unpack[Ts]], *, out: Optional[Tensor[DType, Unpack[Ts]]] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
def cos(
    input: Tensor[DType, Unpack[Ts]], *, out: Optional[Tensor[DType, Unpack[Ts]]] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
def exp(
    input: Tensor[DType, Unpack[Ts]], *, out: Optional[Tensor[DType, Unpack[Ts]]] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def matmul(
    input: Tensor[DType, N1],
    other: Tensor[DType, N1],
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[DType]: ...
@overload
def matmul(
    input: Tensor[DType, Unpack[Rs], N1, N2],
    other: Tensor[DType, Unpack[Qs], N2, N3],
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Qs]]]], N1, N3]: ...
def multinomial(
    input: Tensor[DType, Unpack[Rs], N1],
    num_samples: N2,
    replacement: builtins.bool = ...,
    *,
    generator: Optional[Generator] = ...,
) -> Tensor[DType, Unpack[Rs], N2]: ...
@overload
def unbind(
    input: Tensor[DType, Unpack[Rs], N], dim: L[-1]
) -> Tuple[Tensor[DType, Unpack[Rs]], ...]: ...
@overload
def unbind(
    input: Tensor[DType, N, N1, Unpack[Rs]], dim: L[1]
) -> Tuple[Tensor[DType, N, Unpack[Rs]], ...]: ...
@overload
def unbind(
    input: Tensor[DType, N, Unpack[Rs]], dim: L[0] = ...
) -> Tuple[Tensor[DType, Unpack[Rs]], ...]: ...
@overload
def unsqueeze(
    input: Tensor[DType, Unpack[Ts]], dim: L[-1]
) -> Tensor[DType, Unpack[Ts], L[1]]: ...
@overload
def unsqueeze(
    input: Tensor[DType, Unpack[Ts]], dim: L[0]
) -> Tensor[DType, L[1], Unpack[Ts]]: ...
@overload
def unsqueeze(
    input: Tensor[DType, N, Unpack[Ts]], dim: L[1]
) -> Tensor[DType, N, L[1], Unpack[Ts]]: ...
@overload
def unsqueeze(
    input: Tensor[DType, N1, N2, Unpack[Ts]], dim: L[2]
) -> Tensor[DType, N1, N2, L[1], Unpack[Ts]]: ...
@overload
def real(input: Tensor[complex64, Unpack[Ts]]) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def real(input: Tensor[complex128, Unpack[Ts]]) -> Tensor[float64, Unpack[Ts]]: ...
def zeros_like(
    input: Tensor[DType, Unpack[Ts]],
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def randn(
    size: Tuple[Unpack[Ts]], device: device = ..., requires_grad: builtins.bool = ...
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def randn(
    *size: Unpack[Ts], device: device = ..., requires_grad: builtins.bool = ...
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def all(
    input: Tensor[DType, Unpack[Ts]],
) -> Tensor[bool, L[1]]: ...
@overload
def all(
    input: Tensor[DType, N, Unpack[Ts]],
    dim: L[0],
) -> Tensor[bool, Unpack[Ts]]: ...
@overload
def all(
    input: Tensor[DType, N1, N2, Unpack[Ts]],
    dim: L[1],
) -> Tensor[bool, N1, Unpack[Ts]]: ...
@overload
def randperm(
    n: N,
    *,
    dtype: DType,
    generator: Optional[Generator] = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: _bool = ...,
    requires_grad: _bool = ...,
) -> Tensor[DType, N]: ...
@overload
def randperm(
    n: N,
    *,
    generator: Optional[Generator] = ...,
    out: Optional[Tensor] = ...,
    dtype: Type[float32] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: _bool = ...,
    requires_grad: _bool = ...,
) -> Tensor[float32, N]: ...
def sqrt(
    input: Tensor[DType, Unpack[Ts]], *, out: Optional[Tensor[DType, Unpack[Ts]]] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def where(
    condition: Tensor[torch.bool, Unpack[Ts]],
    x: Tensor[DType, Unpack[Rs]],
    y: Tensor[DType, Unpack[Rs2]],
) -> Tensor[
    DType,
    Unpack[
        Broadcast[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]], Tuple[Unpack[Rs2]]]
    ],
]: ...

# The exact output shape in this case depends on the contents of the tensor,
# meaning this is too dynamic for shape types.
@overload
def where(condition: Tensor[DType, Unpack[Ts]]) -> Any: ...
@overload
def diff(
    input: Tensor[DType, Unpack[Rs], Add[N1, L[1]], N2],
    dim: L[-2],
) -> Tensor[DType, Unpack[Rs], N1, N2]: ...
@overload
def diff(
    input: Tensor[DType, Add[N, L[1]], Unpack[Rs]],
    dim: L[0],
) -> Tensor[DType, N, Unpack[Rs]]: ...
@overload
def diff(
    input: Tensor[DType, N1, Add[N2, L[1]], Unpack[Rs]],
    dim: L[1],
) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
@overload
def diff(
    input: Tensor[DType, Unpack[Rs], Add[N, L[1]]],
    dim: L[-1] = ...,
) -> Tensor[DType, Unpack[Rs], N]: ...
def argsort(
    input: Tensor[DType, Unpack[Ts]], dim: int = ..., descending: builtin_bool = ...
) -> Tensor[DType, Unpack[Ts]]: ...

# Input tuple has 2 elements.
@overload
def cat(
    tensors: Tuple[Tensor[DType, Unpack[Rs], N1], Tensor[DType, Unpack[Rs], N2]],
    dim: L[-1],
    *,
    out: Any = ...,
) -> Tensor[DType, Unpack[Rs], Add[N1, N2]]: ...
@overload
def cat(
    tensors: Tuple[Tensor[DType, N, N1, Unpack[Rs]], Tensor[DType, N, N2, Unpack[Rs]]],
    dim: L[1],
    *,
    out: Any = ...,
) -> Tensor[DType, N, Add[N1, N2], Unpack[Rs]]: ...
@overload
def cat(
    tensors: Tuple[Tensor[DType, N1, Unpack[Rs]], Tensor[DType, N2, Unpack[Rs]]],
    dim: L[0] = ...,
    *,
    out: Any = ...,
) -> Tensor[DType, Add[N1, N2], Unpack[Rs]]: ...

# Input tuple has 3 elements.
@overload
def cat(
    tensors: Tuple[
        Tensor[DType, Unpack[Rs], N1],
        Tensor[DType, Unpack[Rs], N2],
        Tensor[DType, Unpack[Rs], N3],
    ],
    dim: L[-1],
    *,
    out: Any = ...,
) -> Tensor[DType, Unpack[Rs], Add[Add[N1, N2], N3]]: ...
@overload
def cat(
    tensors: Tuple[
        Tensor[DType, N, N1, Unpack[Rs]],
        Tensor[DType, N, N2, Unpack[Rs]],
        Tensor[DType, N, N3, Unpack[Rs]],
    ],
    dim: L[1],
    *,
    out: Any = ...,
) -> Tensor[DType, N, Add[Add[N1, N2], N3], Unpack[Rs]]: ...
@overload
def cat(
    tensors: Tuple[
        Tensor[DType, N1, Unpack[Rs]],
        Tensor[DType, N2, Unpack[Rs]],
        Tensor[DType, N3, Unpack[Rs]],
    ],
    dim: L[0] = ...,
    *,
    out: Any = ...,
) -> Tensor[DType, Add[Add[N1, N2], N3], Unpack[Rs]]: ...

# This takes an arbitrary-length list of tensors and concatenates them across an
# axis. We don't know the length of the list and thus can't tell the final
# dimensions of the tensor.
@overload
def cat(
    tensors: Iterable[Tensor[DType, Unpack[Ts]]], dim: int = ..., *, out: Any = ...
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...

save: Any
manual_seed: Any
load: Any
from_numpy: Any
no_grad: Any

def sign(
    input: Tensor[DType, Unpack[Rs]], *, out: Optional[Tensor] = ...
) -> Tensor[DType, Unpack[Rs]]: ...
def sparse_coo_tensor(
    input: Tensor[DType, Unpack[Rs]], *, out: Optional[Tensor] = ...
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def sparse_coo_tensor(
    indices: Tensor,
    values: Union[Tensor, List[Any]],
    size: Tuple[Unpack[Rs]],
    *,
    dtype: Optional[DType],
    device: Union[_device, str, None] = ...,
    requires_grad: builtin_bool = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def sparse_coo_tensor(
    indices: Tensor,
    values: Union[Tensor, List[Any]],
    size: Tuple[Unpack[Rs]],
    *,
    dtype: L[float32] = ...,
    device: Union[_device, str, None] = ...,
    requires_grad: builtin_bool = ...,
) -> Tensor[float32, Unpack[Rs]]: ...
@overload
def sparse_coo_tensor(
    indices: Tensor,
    values: Union[Tensor, List[Any]],
    size: L[None] = ...,
    *,
    dtype: Type[Any] = ...,
    device: Union[_device, str, None] = ...,
    requires_grad: builtin_bool = ...,
) -> Tensor: ...
@overload
def softmax(
    input: Tensor[DType, Unpack[Ts]], dim: int
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def softmax(
    input: Tensor[DType, Unpack[Ts]], dim: int, dtype: Type[DType2] = ...
) -> Tensor[DType2, Unpack[Ts]]: ...
@overload
def transpose(
    input: Tensor[DType, Unpack[Rs], N1, N2], dim0: L[-2], dim1: L[-1]
) -> Tensor[DType, Unpack[Rs], N2, N1]: ...
@overload
def transpose(
    input: Tensor[DType, Unpack[Rs], N1, N2], dim0: L[-1], dim1: L[-2]
) -> Tensor[DType, Unpack[Rs], N2, N1]: ...
@overload
def transpose(
    input: Tensor[DType, N1, N2, Unpack[Rs]], dim0: L[0], dim1: L[1]
) -> Tensor[DType, N2, N1, Unpack[Rs]]: ...
@overload
def transpose(
    input: Tensor[DType, N1, N2, Unpack[Rs]], dim0: L[1], dim1: L[0]
) -> Tensor[DType, N2, N1, Unpack[Rs]]: ...
@overload
def transpose(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]], dim0: L[1], dim1: L[2]
) -> Tensor[DType, N1, N3, N2, Unpack[Rs]]: ...
@overload
def flatten(
    input: Tensor[DType, N1, Unpack[Rs], N2],
    start_dim: L[0] = ...,
    end_dim: L[-1] = ...,
) -> Tensor[DType, Product[N1, Unpack[Rs], N2]]: ...
@overload
def flatten(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    start_dim: L[0] = ...,
    end_dim: L[1] = ...,
) -> Tensor[DType, Multiply[N1, N2], Unpack[Rs]]: ...
@overload
def flatten(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    start_dim: L[1] = ...,
    end_dim: L[2] = ...,
) -> Tensor[DType, N1, Multiply[N2, N3], Unpack[Rs]]: ...
@overload
def flatten(
    input: Tensor[DType, N1, N2, N3, N4, Unpack[Rs]],
    start_dim: L[2] = ...,
    end_dim: L[3] = ...,
) -> Tensor[DType, N1, N2, Multiply[N3, N4], Unpack[Rs]]: ...
@overload
def flatten(
    input: Tensor[DType],
    start_dim: L[0] = ...,
    end_dim: L[0] = ...,
) -> Tensor[DType, L[1]]: ...
@overload
def reshape(
    input: Tensor[DType, Unpack[Rs]], shape: Tuple[L[-1], Unpack[Rs2]]
) -> Tensor[DType, Divide[Product[Unpack[Rs]], Product[Unpack[Rs2]]], Unpack[Rs2]]: ...
@overload
def reshape(
    input: Tensor[DType, Unpack[Rs]], shape: Tuple[N1, L[-1], Unpack[Rs2]]
) -> Tensor[
    DType, N1, Divide[Product[Unpack[Rs]], Product[N1, Unpack[Rs2]]], Unpack[Rs2]
]: ...
@overload
def reshape(
    input: Tensor[DType, Unpack[Rs]], shape: Tuple[Unpack[Rs2]]
) -> Tensor[DType, Unpack[Rs2]]: ...
