# PyTorch Examples

[![pyre](https://github.com/pradeep90/examples/workflows/Run%20Pyre/badge.svg)](https://github.com/pradeep90/examples/actions/workflows/main.yml)

Example repository showing how to use variadic tuples (PEP 646) to type real-world Tensor code.

NOTE: Right now, we are [focusing](https://github.com/pradeep90/pytorch_examples/blob/master/.pyre_configuration#L3) only on one directory `time_sequence_prediction`. Once we are happy with the stubs for it, we can start looking at other directories as well.

# Catching Tensor Shape Errors using the Typechecker

## TODO

+ TODO: ndarray needs to be generic in the dtype as well.

## Features we will need

+ Matrix multiplication requires a transpose operator.

+ Need some way of extracting a tuple from a list. For example, `tf.zeros([3, 4])` should return `Tensor[float32, L[3], L[4]]`. However, we have no way to extract individual types from a list. Note that we handle a tuple just fine: `tf.zeros((3, 4))`.

  One way is to accept a literal list and have the typechecker convert it to a tuple type.

+ Similarly, we need a way of inferring the shape of nested Python lists. For example, if `xs = [[0 for _ in range(10)]]`, we should be able to infer it has shape `(1, 10)`.

+ Concatenation of multiple variadics

  - `argmax`

+ Broadcasting - this is intended to be part of type arithmetic, but the relevant code has not yet landed in Pyre.

+ Better error messages. They are pretty arcane now.

## Notes on StackOverflow bugs

From https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow

+ UT-2 - tensorflow_program_bugs/ut_2_multiplication.py - this needs broadcasting (from type arithmetic).

+ UT-3 - tensorflow_program_bugs/ut_3_image_set_shape.py - I'm using `tf.ensure_shape` instead of `tf.set_shape`.

  Note that `x.set_shape` can't be typed statically since it changes the type parameters of a Tensor. Basically, `x: Tensor[L[10], L20]; x.set_shape([2, 10, 10])` will make the tensor have shape `2x10x10` at runtime. The Python type system has no way of reflecting the updated `self` type in the return signature. So, I'm using the [TensorFlow-recommended](https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape) `x.ensure_shape` instead, which returns a new Tensor. We can make the returned tensor type have the new shape.

+ UT-4 - this seems to be based on TensorFlow v1 style of `placeholder` code. That seems to be deprecated in v2. I narrowed it down to the parts that caused the shape errors.
