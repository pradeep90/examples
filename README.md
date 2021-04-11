# PyTorch Examples

[![pyre](https://github.com/pradeep90/examples/workflows/Run%20Pyre/badge.svg)](https://github.com/pradeep90/examples/actions/workflows/main.yml)

Example repository showing how to use variadic tuples (PEP 646) to type real-world Tensor code.

NOTE: Right now, we are [focusing](https://github.com/pradeep90/pytorch_examples/blob/master/.pyre_configuration#L3) only on one directory `time_sequence_prediction`. Once we are happy with the stubs for it, we can start looking at other directories as well.

# Catching Tensor Shape Errors using the Typechecker

## Things still TODO:

+ ndarray needs to be generic in the dtype as well.

## Notes on StackOverflow bugs

From https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow

+ UT-2 - tensorflow_program_bugs/ut_2_multiplication.py - this needs broadcasting (from type arithmetic).
