from pyre_extensions import TypeVarTuple
from typing import Any, TypeVar


conv2d: Any
softmax: Any
softmax_cross_entropy_with_logits: Any
sparse_softmax_cross_entropy_with_logits: Any

DType = TypeVar("DType")
Dims = TypeVarTuple("Dims")

def dropout(x: Tensor[DType, *Dims], rate=..., noise_shape=..., seed=..., name=...) -> Tensor[DType, *Dims]: ...

def relu(features: Tensor[DType, *Dims], name: str = ...) -> Tensor[DType, *Dims]: ...

