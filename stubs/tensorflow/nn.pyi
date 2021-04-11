from typing import TypeVar
from tensorflow import (
    ArrayLike,
    Tensor,
)

Batch = TypeVar("Batch", bound=int)
Features = TypeVar("Features", bound=int)
A2 = TypeVar("A2")

def softmax_cross_entropy_with_logits(
    labels: ArrayLike[Batch, Features], logits: Tensor[Batch, Features]
) -> Tensor[Batch]: ...
def sparse_softmax_cross_entropy_with_logits(
    labels: ArrayLike[Batch], logits: Tensor[Batch, Features]
) -> Tensor[Batch]: ...
