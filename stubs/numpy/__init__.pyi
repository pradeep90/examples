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
    Union,
)

from pyre_extensions import TypeVarTuple, Unpack

Ts = TypeVarTuple("Ts")
Ts2 = TypeVarTuple("Ts2")

_Shape = Tuple[Unpack[Ts]]

class _ArrayOrScalarCommon(
    Generic[Unpack[Ts]],
    SupportsInt,
    SupportsFloat,
    SupportsComplex,
    SupportsBytes,
    SupportsAbs[Any],
): ...

class ndarray(_ArrayOrScalarCommon[Unpack[Ts]], Iterable, Sized, Container):
    def __init__(
        self,
        shape: Tuple[Unpack[Ts]],
        dtype=...,
        buffer=...,
        offset: Optional[int] = ...,
        strides: Tuple[int, ...] = ...,
        order: Optional[str] = ...,
    ) -> None: ...
    def __setitem__(self, key, value): ...
    @property
    def shape(self) -> Tuple[Unpack[Ts]]: ...
    @overload
    def reshape(self, shape: Tuple[Unpack[Ts2]]) -> ndarray[Unpack[Ts2]]: ...
    @overload
    def reshape(self, *shape: Unpack[Ts2]) -> ndarray[Unpack[Ts2]]: ...
    def __add__(self, other) -> ndarray[Unpack[Ts]]: ...
    def __div__(self, other) -> ndarray[Unpack[Ts]]: ...
    def __truediv__(self, other) -> ndarray[Unpack[Ts]]: ...
    def astype(self, dtype: str) -> ndarray[Unpack[Ts]]: ...

def empty(
    shape: Union[int, Tuple[Unpack[Ts]]], dtype: Union[Type[Any], str]
) -> ndarray[Unpack[Ts]]: ...
def array(
    object: object,
    dtype=...,
    copy: bool = ...,
    subok: bool = ...,
    ndmin: int = ...,
) -> ndarray[Unpack[Tuple[Any, ...]]]: ...
def sin(x: ndarray[Unpack[Ts]]) -> ndarray[Unpack[Ts]]: ...
