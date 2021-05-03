from typing_extensions import Literal

import numpy as np
import tensorflow as tf

L2 = Literal[2]
L3 = Literal[3]
L5 = Literal[5]

matrix1 = tf.Variable(np.random.randn(5, 2))
matrix2 = tf.Variable(np.random.randn(2, 2, 3))

# This is what you'd _think_ the shape would be:
result: tf.Tensor[tf.float32, L5, L2, L3] = tf.matmul(matrix1, matrix2)  # pyre-fixme[9]
# But `matmul` is weird, and the result is _actually_:
result2: tf.Tensor[tf.float32, L2, L5, L3] = tf.matmul(matrix1, matrix2)
