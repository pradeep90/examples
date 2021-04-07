from pyre_extensions import TypeVarTuple, Unpack
from typing import TypeVar
from . import ndarray

Ts = TypeVarTuple("Ts")

N = TypeVar("N", bound=int)

def randn(*args: Unpack[Ts]) -> ndarray[Unpack[Ts]]: ...
def randint(low: N, high: N = ..., size=None, dtype=int) -> ndarray[N]: ...

seed: Any = ...
