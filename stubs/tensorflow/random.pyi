from typing import Tuple, TypeVar
from tensorflow import float32, Tensor
from pyre_extensions import TypeVarTuple, Unpack

Ts = TypeVarTuple("Ts")
DType = TypeVar("DType")

def truncated_normal(shape: Tuple[Unpack[Ts]], stddev: float) -> Tensor[float32, Unpack[Ts]]: ...
def uniform(
    shape: Tuple[Unpack[Ts]],
    minval: Optional[Union[int, float]] = ...,
    maxval: Optional[Union[int, float]] = ...,
    dtype: Optional[Any] = ...,
    seed: Optional[int] = ...,
    name: Optional[str] = ...,
) -> Tensor[float32, Unpack[Ts]]: ...
