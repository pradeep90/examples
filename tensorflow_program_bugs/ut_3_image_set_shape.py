# UT-3 from
# https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow/UT-3/35451948-buggy/image_set_shape.py

import tensorflow as tf
import numpy as np
from typing_extensions import Literal as L
from typing import Any

x = tf.zeros((3, 3))
y = tf.ensure_shape(x, (3, 3))
# pyre-ignore[6]: Incompatible parameter type.
z = tf.ensure_shape(x, (1, 2, 3))
