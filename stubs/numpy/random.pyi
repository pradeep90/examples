from pyre_extensions import TypeVarTuple, Unpack
from typing import Any, TypeVar, Type, Tuple, overload
from . import ndarray, int64, float32, float64

Ts = TypeVarTuple("Ts")
N = TypeVar("N", bound=int)
DType = TypeVar("DType")

Number = Union[int, float]

def randn(*args: Unpack[Ts]) -> ndarray[float64, Unpack[Ts]]: ...

# ===== BEGIN `randint` =====
# `dtype` not specified
@overload
def randint(
    low: int, high: int = ..., size: Tuple[Unpack[Ts]] = ...
) -> ndarray[int64, Unpack[Ts]]: ...
@overload
def randint(
    low: int, high: int = ..., size: N = ...
) -> ndarray[int64, N]: ...
# `dtype` specified
@overload
def randint(
    low: int, high: int = ..., size: Tuple[Unpack[Ts]] = ..., dtype: Type[DType] = ...
) -> ndarray[DType, Unpack[Ts]]: ...
@overload
def randint(
    low: int, high: int = ..., size: N = ..., dtype: Type[DType] = ...
) -> ndarray[DType, N]: ...
# ===== END `randint` ======

# ===== BEGIN `normal` =====
@overload
def normal(loc: Number, scale: Number, size: None = None) -> float: ...
@overload
def normal(
    loc: Number, scale: Number, size: Tuple[Unpack[Ts]]
) -> ndarray[float64, Unpack[Ts]]: ...
# ===== END `normal` =====

seed: Any = ...
uniform: Any
