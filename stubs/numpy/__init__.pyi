from typing import (
    Any,
    Container,
    Generic,
    Iterable,
    Sized,
    SupportsInt,
    SupportsFloat,
    SupportsComplex,
    SupportsBytes,
    SupportsAbs,
    overload,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pyre_extensions import TypeVarTuple, Unpack

DType = TypeVar("DType")
Ts = TypeVarTuple("Ts")
Ts2 = TypeVarTuple("Ts2")

_Shape = Tuple[Unpack[Ts]]

class _ArrayOrScalarCommon(
    Generic[DType, Unpack[Ts]],
    SupportsInt,
    SupportsFloat,
    SupportsComplex,
    SupportsBytes,
    SupportsAbs[Any],
): ...
class float: ...

class ndarray(_ArrayOrScalarCommon[DType, Unpack[Ts]], Iterable, Sized, Container):
    def __init__(
        self,
        shape: Tuple[Unpack[Ts]],
        dtype: Type[DType] = ...,
        buffer=...,
        offset: Optional[int] = ...,
        strides: Tuple[int, ...] = ...,
        order: Optional[str] = ...,
    ) -> None: ...
    def __setitem__(self, key, value): ...
    @property
    def shape(self) -> Tuple[Unpack[Ts]]: ...
    @overload
    def reshape(self, shape: Tuple[Unpack[Ts2]]) -> ndarray[DType, Unpack[Ts2]]: ...
    @overload
    def reshape(self, *shape: Unpack[Ts2]) -> ndarray[DType, Unpack[Ts2]]: ...
    def __add__(self, other) -> ndarray[DType, Unpack[Ts]]: ...
    def __div__(self, other) -> ndarray[DType, Unpack[Ts]]: ...
    def __truediv__(self, other) -> ndarray[DType, Unpack[Ts]]: ...
    def astype(self, dtype: str) -> ndarray[DType, Unpack[Ts]]: ...

def empty(
    shape: Union[int, Tuple[Unpack[Ts]]], dtype: Union[Type[DType], str]
) -> ndarray[DType, Unpack[Ts]]: ...
def array(
    object: object,
    dtype: Type[DType] = ...,
    copy: bool = ...,
    subok: bool = ...,
    ndmin: int = ...,
) -> ndarray[DType, Unpack[Tuple[Any, ...]]]: ...
def sin(x: ndarray[DType, Unpack[Ts]]) -> ndarray[DType, Unpack[Ts]]: ...
