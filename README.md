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


### UT-12

The actual issue here is that the programmer tried to flatten a tensor using `reshape(h, [-1, 4 * 4 * 64])`. Unfortunately, they got the size wrong - the tensor was actually shape `batch x 8 x 8 x 64`. This doesn't actually cause an error until later in the code, when the incorrect flattening causes a later `reshape` to result in a tensor with the wrong batch size.

While we _could_ statically detect the same error as is produced at runtime, I'm not sure that would be very interesting - that error doesn't help in narrowing down what the problem really is. And I don't think we can detect the problem with the reshape with static analysis, because the programmer simply told the program to do the wrong thing.

The kind of thing that would _actually_ help the programmer here would either be:

* A system which shows statically-inferred shapes to the user _while_ they're coding.
* Type annotations specifying what the user expects the shape after each layer to be, which could be checked _after_ the user has finished coding.

Unfortunately, both of these involve complicated calculations to determine how convolutions and pooling change the shape, which we can't do without type arithmetic.

So yeah, however we slice this one, I don't think we can do anything useful :(
