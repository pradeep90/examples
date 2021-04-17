import numpy as np
from typing import Optional, overload, TypeVar, Union
from tensorflow import int32, int64, Tensor

DType = TypeVar("DType")
Batch = TypeVar("Batch", bound=int)
Features = TypeVar("Features", bound=int)

# ===== BEGIN `softmax_cross_entropy_with_logits` =====
# `ndarray` for `labels`
@overload
def softmax_cross_entropy_with_logits(
    labels: np.ndarray[Union[np.int32, np.int64], Batch, Features], logits: Tensor[DType, Batch, Features]
) -> Tensor[DType, Batch]: ...
# `Tensor` for `labels
@overload
def softmax_cross_entropy_with_logits(
    labels: Tensor[Union[int32, int64], Batch, Features], logits: Tensor[DType, Batch, Features]
) -> Tensor[DType, Batch]: ...
# ===== END `softmax_cross_entropy_with_logits` =====
# ===== BEGIN `softmax_cross_entropy_with_logits` =====
@overload
def sparse_softmax_cross_entropy_with_logits(
    labels: np.ndarray[Union[np.int32, np.int64], Batch], logits: Tensor[DType, Batch, Features]
) -> Tensor[DType, Batch]: ...
@overload
def sparse_softmax_cross_entropy_with_logits(
    labels: Tensor[Union[int32, int64], Batch], logits: Tensor[DType, Batch, Features]
) -> Tensor[DType, Batch]: ...
# ===== END `softmax_cross_entropy_with_logits` =====
def relu(
    features: Tensor[DType, Unpack[Ts]], name: Optional[str] = None
) -> Tensor[DType, Unpack[Ts]]: ...
def dropout(x: Tensor[DType, Unpack[Ts]], rate: float) -> Tensor[DType, Unpack[Ts]]: ...
