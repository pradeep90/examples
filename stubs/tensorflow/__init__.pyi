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
import numpy as np

Ts = TypeVarTuple("Ts")
NewShape = TypeVarTuple("NewShape")
T = TypeVar("T")
DType = TypeVar("DType")

A1 = TypeVar("A1")
A2 = TypeVar("A2")
A3 = TypeVar("A3")
A4 = TypeVar("A4")
A5 = TypeVar("A5")

N = TypeVar("N", bound=int)

TensorLike = Union[
    Tensor[DType, Unpack[Ts]],
    Variable[DType, Unpack[Ts]],
]

class Variable(Generic[DType, Unpack[Ts]]):
    shape: Tuple[Unpack[Ts]]
    # If `initial_value` is a `TensorLike`, then we can link its
    # `DType` to the `DType` of the variable.
    @overload
    def __init__(
        self,
        initial_value: TensorLike[DType, Unpack[Ts]] = ...,
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
    # If `initial_value` is an `ndarray`, though, since NumPy
    # uses different data types to TensorFlow, we should _not_
    # link its `DType` to the `DType` of the `Variable`.
    @overload
    def __init__(
        self,
        initial_value: np.ndarray[Any, Unpack[Ts]] = ...,
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

class float64: ...

class Tensor(Generic[DType, Unpack[Ts]]):
    shape: Tuple[Unpack[Ts]]
    def __eq__(self, other: Tensor[DType, Unpack[Ts]]) -> bool: ...
    def __add__(self: Tensor[DType, A1, A2], other: ArrayLike[DType, A2]) -> Tensor[DType, A1, A2]: ...
    def __truediv__(self, other: int) -> Tensor[DType, Unpack[Ts]]: ...

# ===== BEGIN `matmul` =====

# Normal matrix multiplication: (A1, A2) x (A2, A3).
@overload
def matmul(
    a: TensorLike[DType, A1, A2],
    b: TensorLike[DType, A2, A3],
    transpose_a=...,
    transpose_b=...,
    adjoint_a=...,
    adjoint_b=...,
    a_is_sparse=...,
    b_is_sparse=...,
    name=...,
) -> Tensor[DType, A1, A3]: ...
@overload
def matmul(
    a: np.ndarray[np.float32, A1, A2],
    b: TensorLike[float32, A2, A3],
    transpose_a=...,
    transpose_b=...,
    adjoint_a=...,
    adjoint_b=...,
    a_is_sparse=...,
    b_is_sparse=...,
    name=...,
) -> Tensor[float32, A1, A3]: ...

# (A2, A3) x (A3, A4) with a batch size of A1.
@overload
def matmul(
    a: TensorLike[DType, A1, A2, A3],
    b: TensorLike[DType, A1, A3, A4],
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
    a: TensorLike[DType, A1, A2],
    b: TensorLike[DType, A4, A2, A3],
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
def one_hot(indices: np.ndarray[np.int32, A1], depth: N) -> Tensor[float32, A1, N]: ...
def reduce_sum(x: Tensor[DType, Unpack[Ts]]) -> Tensor[DType]: ...
def pow(x: Tensor[DType, Unpack[Ts]], y: int) -> Tensor[DType, Unpack[Ts]]: ...
def reshape(
    tensor: Tensor[DType, Unpack[Ts]], shape: Tuple[Unpack[NewShape]]
) -> Tensor[DType, Unpack[NewShape]]: ...
def constant(value: float, shape: Tuple[Unpack[Ts]]) -> Tensor[float32, Unpack[Ts]]: ...

GradientTape = Any
reduce_mean = Any
function: Any
