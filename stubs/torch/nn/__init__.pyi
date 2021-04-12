from pyre_extensions import TypeVarTuple, Unpack

from typing import (
    Any,
    Generic,
    Tuple,
    TypeVar,
)

from .. import Tensor

DType = TypeVar("DType")
Ts = TypeVarTuple("Ts")
InputSize = TypeVar("InputSize", bound=int)
OutputSize = TypeVar("OutputSize", bound=int)
HiddenSize = TypeVar("HiddenSize", bound=int)
Batch = TypeVar("Batch", bound=int)
N = TypeVar("N", bound=int)

class Module:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class LSTMCell(Module, Generic[InputSize, HiddenSize]):
    def __init__(
        self, input_size: InputSize, hidden_size: HiddenSize, bias: bool = ...
    ) -> None: ...
    def __call__(
        self,
        input: Tensor[DType, Batch, InputSize],
        hidden: Tuple[
            Tensor[DType, Batch, HiddenSize], Tensor[DType, Batch, HiddenSize]
        ] = ...,
    ) -> Tuple[Tensor[DType, Batch, HiddenSize], Tensor[DType, Batch, HiddenSize]]: ...

class Linear(Module, Generic[InputSize, OutputSize]):
    def __init__(
        self, in_features: InputSize, out_features: OutputSize, bias: bool = ...
    ) -> None: ...
    def __call__(
        self,
        input: Tensor[DType, N, Unpack[Ts], InputSize],
    ) -> Tensor[DType, N, Unpack[Ts], OutputSize]: ...

def __getattr__(name) -> Any: ...
