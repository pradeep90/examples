from typing import Any, Generic, Optional, overload, NoReturn, Tuple, TypeVar
from pyre_extensions import TypeVarTuple

Shape = TupleVarTuple('Shape')
A1 = TypeVar('A1')
A2 = TypeVar('A2')
A3 = TypeVar('A3')


class Tensor(Generic[*Shape]):
  ...


def placeholder(
    dtype,
    shape: Tuple[int],
) -> Tensor[Any]:
  ...


def ones(
    shape: Tuple[int, int],
) -> Tensor[Any]:
  ...


def Variable(
    initial_value: Tensor[A1],
) -> Tensor[A1]:
  ...


class Session:

    def __enter__(self) -> Session:
        ...

    def run(self, x: Tensor, feed_dict=None):
        ...
 

# ===== matul =====

@overload
def matmul(
    a: Tensor[A1],
    b: Tensor[A1],
) -> Tensor[A1]:
  ...

@overload
def matmul(
    a: Tensor[A1],
    b: Tensor[A1, A2],
) -> Tensor[A2]:
  ...

@overload
def matmul(
    a: Tensor[A1, A2],
    b: Tensor[A2],
) -> Tensor[A1]:
  ...

@overload
def matmul(
    a: Tensor[A1, A2],
    b: Tensor[A2, A3],
) -> Tensor[A1, A3]:
  ...

@overload
def matmul(
    a: Any,
    b: Any,
) -> NoReturn:
  ...
