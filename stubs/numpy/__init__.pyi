from typing import (
    Any,
    Container,
    Generic,
    Iterable,
    Optional,
    overload,
    Sized,
    SupportsAbs,
    SupportsBytes,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# pyre-ignore[21]: Could not find module `pyre_extensions`. (Spurious error)
from pyre_extensions import Broadcast, Divide, Product, TypeVarTuple, Unpack
from typing_extensions import Literal as L

DType = TypeVar("DType")
NewDType = TypeVar("NewDType")
Ts = TypeVarTuple("Ts")
Rs = TypeVarTuple("Rs")
Qs = TypeVarTuple("Qs")
Ts2 = TypeVarTuple("Ts2")

N = TypeVar("N")
N1 = TypeVar("N1")
N2 = TypeVar("N2")
N3 = TypeVar("N3")
A1 = TypeVar("A1")
A2 = TypeVar("A2")

class _ArrayOrScalarCommon(
    Generic[DType, Unpack[Ts]],
    SupportsInt,
    SupportsFloat,
    SupportsComplex,
    SupportsBytes,
    SupportsAbs[Any],
): ...
class float: ...
class int: ...

class ndarray(Generic[DType, Unpack[Ts]]):
    def __init__(
        self,
        shape: Tuple[Unpack[Ts]],
        dtype: Type[DType] = ...,
        buffer=...,
        offset: Optional[int] = ...,
        strides: Tuple[int, ...] = ...,
        order: Optional[str] = ...,
    ) -> None: ...
    @overload
    def __getitem__(self: ndarray[DType, A1, A2], key: L[0]) -> ndarray[DType, A2]: ...
    @overload
    def __getitem__(self: ndarray[DType, A1, A2], key: L[1]) -> ndarray[DType, A1]: ...
    def __setitem__(self, key, value): ...
    @property
    def shape(self) -> Tuple[Unpack[Ts]]: ...
    @property
    def T(self: ndarray[DType, N1, N2]) -> ndarray[DType, N2, N1]: ...
    @overload
    def __add__(
        self, other: ndarray[DType, Unpack[Rs]]
    ) -> ndarray[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    def __div__(self, other) -> ndarray[DType, Unpack[Ts]]: ...
    def __truediv__(self, other) -> ndarray[DType, Unpack[Ts]]: ...
    def __matmul__(
        self: ndarray[DType, Unpack[Rs], N1, N2],
        other: ndarray[DType, Unpack[Qs], N2, N3],
    ) -> ndarray[
        DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Qs]]]], N1, N3
    ]: ...
    @overload
    def reshape(
        self, *shape: Unpack[Tuple[Unpack[Rs], L[-1]]]
    ) -> ndarray[
        DType, Unpack[Rs], Divide[Product[Unpack[Ts]], Product[Unpack[Rs]]]
    ]: ...
    @overload
    def reshape(self, *shape: Unpack[Rs]) -> ndarray[DType, Unpack[Rs]]: ...
    @overload
    def sum(
        self: ndarray[DType, N1, Unpack[Rs]],
        axis: L[0],
    ) -> ndarray[DType, Unpack[Rs]]: ...

    # ===== BEGIN `astype` =====
    @overload
    def astype(self, dtype: Type[NewDType]) -> ndarray[NewDType, Unpack[Ts]]: ...
    @overload
    def astype(self, dtype: L["int64"]) -> ndarray[int64, Unpack[Ts]]: ...
    @overload
    def astype(self, dtype: L["float32"]) -> ndarray[float32, Unpack[Ts]]: ...
    @overload
    def astype(self, dtype: L["float64"]) -> ndarray[float64, Unpack[Ts]]: ...
    # ===== END `astype` =====

# ===== BEGIN `empty` =====
# `shape` as tuple, dtype="int64"
@overload
def empty(
    shape: Tuple[Unpack[Ts]], dtype: L["int64"]
) -> ndarray[int64, Unpack[Ts]]: ...

# `shape` as tuple, dtype as e.g. np.float32
@overload
def empty(
    shape: Tuple[Unpack[Ts]], dtype: Type[DType]
) -> ndarray[DType, Unpack[Ts]]: ...

# `shape` as integer, dtype as e.g. np.float32
@overload
def empty(shape: N, dtype: Type[DType]) -> ndarray[DType, N]: ...

# ===== END `empty` =====
def array(
    object: object,
    dtype: Type[DType] = ...,
    copy: bool = ...,
    subok: bool = ...,
    ndmin: int = ...,
) -> ndarray[DType, Unpack[Tuple[Any, ...]]]: ...
def sin(x: ndarray[DType, Unpack[Ts]]) -> ndarray[DType, Unpack[Ts]]: ...

class int64:
    def __init__(self, value=...): ...

class float32:
    def __init__(self, value=...): ...

class float64:
    def __init__(self, value=...): ...

loadtxt: Any
asarray: Any

@overload
def zeros(
    size: Tuple[Unpack[Ts]], dtype: Type[DType]
) -> ndarray[DType, Unpack[Ts]]: ...
@overload
def zeros(
    size: Tuple[Unpack[Ts]], dtype: Type[float] = ...
) -> ndarray[float, Unpack[Ts]]: ...
@overload
def arange(
    stop: N,
    *,
    dtype: Optional[Type[int]] = ...,
) -> ndarray[int, N]: ...
