"""UT-8

Adapted to TensorFlow 2 from
https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow/UT-8/34908033-buggy/multiply.py
which, in turn, is originally from
https://stackoverflow.com/questions/34908033/valueerror-when-performing-matmul-with-tensorflow
"""

import tensorflow as tf


a = tf.ones((2, 2))

# Broken:
b = tf.ones((2,))
c = tf.matmul(a, b)  # pyre-fixme[6]
# TensorFlow says:
#  InvalidArgumentError: In[0] and In[1] has different ndims: [2,2] vs. [2] [Op:MatMul]
# Pyre says:
#  Incompatible parameter type [6]: Expected
#  `tf.Tensor[Variable[tf.T], Variable[tf.A2], Variable[tf.A3]]`
#  for 2nd positional only parameter to call `tf.matmul` but got
#  `tf.Tensor[tf.float32, int]`.

# Correct:
b = tf.ones((2, 1))
c = tf.matmul(a, b)
# With `reveal_type(c)`, Pyre says:
#  Revealed type for `c` is `tf.Tensor[tf.float32, typing_extensions.Literal[2], typing_extensions.Literal[1]]`.
