Matrix multiplication.

From [TensorFlow-Program-Bugs/StackOverflow/UT-2](https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow/UT-2/43067338-buggy/multiplication.py), based on [stackoverflow/43067338](https://stackoverflow.com/questions/43067338/tensor-multiplication-in-tensorflow).

This produces a runtime error because of incorrect arguments to `tf.matmul`. We can predict this error with type checking 'for free' - that is, without having to add any extra type annotations to the code (assuming we shape-type stubs for TensorFlow).
