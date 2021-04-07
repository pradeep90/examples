from typing import Generic, Optional, Override, NoReturn, Tuple, TypeVar
from pyre_extensions import TypeVarTuple

Shape = TupleVarTuple('Shape')
A1 = TypeVar('A1')
A2 = TypeVar('A2')
A3 = TypeVar('A3')

class Tensor(Generic[*Shape]):
  ..

def placeholder(
    dtype,
    shape: Tuple[int],
) -> Tensor[Any, Any]:
  ...

def ones(
    shape: Tuple[int, int],
) -> Tensor[Any]:
  ...

def Variable(
    initial_value: Tensor[A1],
) -> Tensor[A1]:
  ...
 
# ===== matul =====

@override
def matmul(
    a: Tensor[A1],
    b: Tensor[A1],
) -> Tensor[A1]:
  ...

@override
def matmul(
    a: Tensor[A1],
    b: Tensor[A1, A2],
) -> Tensor[A2]:
  ...

@override
def matmul(
    a: Tensor[A1, A2],
    b: Tensor[A2],
) -> Tensor[A1]:
  ...

@override
def matmul(
    a: Tensor[A1, A2],
    b: Tensor[A2, A3],
) -> Tensor[A1, A3]:
  ...

@override
def matmul(
    a: Any,
    b: Any,
) -> NoReturn:
  ...
