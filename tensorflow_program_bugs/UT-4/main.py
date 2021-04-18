import tensorflow as tf
import numpy as np
from typing_extensions import Literal as L
from typing import Any
from pyre_extensions import Divide, TypeVarTuple, Unpack

from typing import TypeVar, Tuple, overload

Ts = TypeVarTuple("Ts")
N = TypeVar("N", bound=int)

# I've narrowed it down to the essential part of the code.

NUM_CLASSES = 10
NUM_ROWS = 50


def generate_unit_test(
    length: N,
) -> Tuple[tf.Tensor[tf.float32, N, L[56], L[56], L[3]], tf.Tensor[tf.float32, N]]:
    return (
        tf.ones((length, 56, 56, 3)),
        tf.ones((length,)),
    )


def buggy() -> None:
    x, y = generate_unit_test(NUM_ROWS)
    actual = tf.ones((NUM_ROWS, NUM_CLASSES))
    actual_prediction = tf.math.argmax(actual, 1)

    # pyre-ignore: Can't argmax across axis 1 because y has shape (NUM_ROWS,).
    expected_prediction = tf.math.argmax(y, 1)

    # pyre-ignore: `==` not supported for tensors of different shapes.
    correct_prediction = actual_prediction == expected_prediction
