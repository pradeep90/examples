from pyre_extensions import TypeVarTuple, Unpack
from typing import Any, TypeVar, Type, Tuple, overload
from . import ndarray, float

Ts = TypeVarTuple("Ts")
N = TypeVar("N", bound=int)
DType = TypeVar("DType")

Number = Union[int, float]

def randn(*args: Unpack[Ts]) -> ndarray[float, Unpack[Ts]]: ...
@overload
def randint(
    low: int, high: int = ..., size: Tuple[Unpack[Ts]] = ..., dtype: Type[DType] = ...
) -> ndarray[DType, Unpack[Ts]]: ...
@overload
def randint(
    low: int, high: int = ..., size: N = ..., dtype: Type[int] = ...
) -> ndarray[int, N]: ...

# ===== Begin normal =====
@overload
def normal(loc: Number, scale: Number, size: None = None) -> float: ...
@overload
def normal(
    loc: Number, scale: Number, size: Tuple[Unpack[Ts]]
) -> ndarray[np.float32, Unpack[Ts]]: ...

# ===== End normal =====

seed: Any = ...
