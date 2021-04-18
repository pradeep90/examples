An attempt at an MNIST classifier.

From [TensorFlow-Program-Bugs/StackOverflow/UT-1](https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow/UT-1/38167455-buggy/mnist.py), based on [stackoverflow/38167455](https://stackoverflow.com/questions/38167455/tensorflow-output-from-stride).

This code produces a runtime error when trying to flatten `h_pool2`.
This is because the programmer thought the size of `h_pool2` would be `7 * 7 * 64`, when actually it's `4 * 4 * 64` because of the use of `padding='VALID'` when performing convolutions. The fix is to switch to `padding='SAME'`.

We can't detect this same error with shape-type checking (yet), but with some extra annotations (`main_typed.py`) we can at least use type checking to infer what the shape should be after each layer. This could be useful to a programmer debugging this using an IDE that shows inferred types when hovering over each variable.

...except we can't currently, because for some reason Pyre can't infer the type of e.g. `W_conv1` within `compute_prediction`. TODO: Investigate and fix.
