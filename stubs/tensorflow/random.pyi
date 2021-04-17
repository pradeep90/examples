from typing import Tuple, TypeVar
from tensorflow import float32, Tensor
from pyre_extensions import TypeVarTuple, Unpack

Ts = TypeVarTuple("Ts")
DType = TypeVar("DType")

def truncated_normal(shape: Tuple[Unpack[Ts]], stddev: float) -> Tensor[float32, Unpack[Ts]]: ...
