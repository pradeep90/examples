from pyre_extensions import TypeVarTuple, Unpack
from typing import Any, Generic, TypeVar, Tuple, overload

from .nn import nn as nn

DType = TypeVar("DType")
Ts = TypeVarTuple("Ts")

class float32: ...
class Tensor(Generic[DType, Unpack[Ts]]): ...

save: Any
manual_seed: Any
load: Any
from_numpy: Any
no_grad: Any
