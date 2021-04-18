from pyre_extensions import TypeVarTuple, Unpack, Add, Multiply, Divide
from typing_extensions import Literal

from typing import (
    Any,
    Generic,
    Iterator,
    Tuple,
    TypeVar,
)

from .. import Tensor

DType = TypeVar("DType")
T = TypeVar("T")
Ts = TypeVarTuple("Ts")
InputSize = TypeVar("InputSize", bound=int)
OutputSize = TypeVar("OutputSize", bound=int)
HiddenSize = TypeVar("HiddenSize", bound=int)
Batch = TypeVar("Batch", bound=int)
N = TypeVar("N", bound=int)

class Module:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def parameters(self) -> Iterator[Any]: ...
    def double(self: T) -> T: ...

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

class _Loss(Module): ...

class MSELoss(_Loss):
    def __call__(
        self,
        input: Tensor[DType, N, Unpack[Ts]],
        target: Tensor[DType, N, Unpack[Ts]],
    ) -> Tensor[DType]: ...

InChannels = TypeVar("InChannels", bound=int)
OutChannels = TypeVar("OutChannels", bound=int)
KernelSize = TypeVar("KernelSize", bound=int)
Stride = TypeVar("Stride", bound=int)
Batch = TypeVar("Batch", bound=int)
Height = TypeVar("Height", bound=int)
Width = TypeVar("Width", bound=int)
Channels = TypeVar("Channels", bound=int)
Padding = TypeVar("Padding", bound=int)

class Conv2d(Generic[InChannels, OutChannels, KernelSize, Stride]):
    def __init__(
        self,
        in_channels: InChannels,
        out_channels: OutChannels,
        kernel_size: KernelSize,
        stride: Stride,
    ) -> None: ...
    def __call__(
        self, input: Tensor[DType, Batch, InChannels, Height, Width]
    ) -> Tensor[
        DType,
        Batch,
        OutChannels,
        # [(W−K+2P) / S] + 1.
        Add[Divide[Add[Height, Multiply[Literal[-1], KernelSize]], Stride], Literal[1]],
        Add[Divide[Add[Width, Multiply[Literal[-1], KernelSize]], Stride], Literal[1]],
    ]: ...

class ReflectionPad2d(Generic[Padding]):
    def __init__(
        self,
        padding: Padding,
    ) -> None: ...
    def __call__(
        self,
        input: Tensor[DType, Batch, Channels, Height, Width],
    ) -> Tensor[
        DType,
        Batch,
        Channels,
        Add[Add[Height, Padding], Padding],
        Add[Add[Width, Padding], Padding],
    ]: ...

class InstanceNorm2d(Generic[Channels]):
    def __init__(self, num_features: Channels, affine: bool = False) -> None: ...
    def __call__(
        self, input: Tensor[DType, Batch, Channels, Height, Width]
    ) -> Tensor[DType, Batch, Channels, Height, Width]: ...

class ReLU:
    def __call__(
        self, input: Tensor[DType, Batch, Channels, Height, Width]
    ) -> Tensor[DType, Batch, Channels, Height, Width]: ...

def __getattr__(name) -> Any: ...
