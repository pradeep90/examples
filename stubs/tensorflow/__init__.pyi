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
    Literal as L,
    Tuple,
    TypeVar,
    Type,
    Union,
)

from pyre_extensions import TypeVarTuple, Unpack
from numpy import ndarray

Ts = TypeVarTuple("Ts")
T = TypeVar("T")
DType = TypeVar("DType")

A1 = TypeVar("A1")
A2 = TypeVar("A2")
A3 = TypeVar("A3")
A4 = TypeVar("A4")
A5 = TypeVar("A5")

N = TypeVar("N", bound=int)

ArrayLike = Union[
    ndarray[DType, Unpack[Ts]],
    Tensor[Any, Unpack[Ts]],
    Variable[Any, Unpack[Ts]],
]

class Variable(Generic[DType, Unpack[Ts]]):
    shape: Tuple[Unpack[Ts]]
    def __init__(
        self,
        initial_value: ArrayLike[DType, Unpack[Ts]] = ...,
        trainable=...,
        validate_shape=...,
        caching_device=...,
        name=...,
        variable_def=...,
        dtype: Type[DType] = ...,
        import_scope=...,
        constraint=...,
        synchronization=...,
        aggregation=...,
        shape=...,
    ) -> None: ...

Session: Any = ...
global_variables_initializer: Any = ...
seed: Any = ...

class float32: ...

class Tensor(Generic[DType, Unpack[Ts]]):
    shape: Tuple[Unpack[Ts]]
    def __eq__(self, other: Tensor[DType, Unpack[Ts]]) -> bool: ...
    def __add__(self: Tensor[A1, A2], other: ArrayLike[A2]) -> Tensor[A1, A2]: ...

# ===== BEGIN `matmul` =====

# Normal matrix multiplication: (A1, A2) x (A2, A3).
@overload
def matmul(
    a: ArrayLike[DType, A1, A2],
    b: ArrayLike[DType, A2, A3],
    transpose_a=...,
    transpose_b=...,
    adjoint_a=...,
    adjoint_b=...,
    a_is_sparse=...,
    b_is_sparse=...,
    name=...,
) -> Tensor[DType, A1, A3]: ...

# (A2, A3) x (A3, A4) with a batch size of A1.
@overload
def matmul(
    a: ArrayLike[DType, A1, A2, A3],
    b: ArrayLike[DType, A1, A3, A4],
    transpose_a=...,
    transpose_b=...,
    adjoint_a=...,
    adjoint_b=...,
    a_is_sparse=...,
    b_is_sparse=...,
    name=...,
) -> Tensor[DType, A1, A2, A4]: ...

# (A1, A2) x (A2, A3), with a batch size of A4,
# with first arg converted to batch size A4 through broadcasting.
@overload
def matmul(
    a: ArrayLike[DType, A1, A2],
    b: ArrayLike[DType, A4, A2, A3],
    transpose_a=...,
    transpose_b=...,
    adjoint_a=...,
    adjoint_b=...,
    a_is_sparse=...,
    b_is_sparse=...,
    name=...,
) -> Tensor[DType, A4, A1, A3]: ...

# ===== END `matmul` =====
@overload
def zeros(
    shape: Tuple[Unpack[Ts]], dtype: Type[T], name: str = ...
) -> Tensor[T, Unpack[Ts]]: ...
@overload
def zeros(
    shape: Tuple[Unpack[Ts]], dtype: Type[float32] = ..., name: str = ...
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def ones(
    shape: Tuple[Unpack[Ts]], dtype: Type[T], name: str = ...
) -> Tensor[T, Unpack[Ts]]: ...
@overload
def ones(
    shape: Tuple[Unpack[Ts]], dtype: Type[float32] = ..., name: str = ...
) -> Tensor[float32, Unpack[Ts]]: ...
def ensure_shape(
    x: Tensor[T, Unpack[Ts]], shape: Tuple[Unpack[Ts]], name: str = ...
) -> Tensor[T, Unpack[Ts]]: ...
def one_hot(indices: ArrayLike[A1], depth: N) -> Tensor[Any, A1, N]: ...

GradientTape = Any
reduce_mean = Any
