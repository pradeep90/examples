# PyTorch Examples

[![pyre](https://github.com/pradeep90/examples/workflows/Run%20Pyre/badge.svg)](https://github.com/pradeep90/examples/actions/workflows/main.yml)

Example repository showing how to use variadic tuples (PEP 646) to type real-world Tensor code.

NOTE: Right now, we are [focusing](https://github.com/pradeep90/pytorch_examples/blob/master/.pyre_configuration#L3) only on one directory `time_sequence_prediction`. Once we are happy with the stubs for it, we can start looking at other directories as well.

# Catching Tensor Shape Errors using the Typechecker

## TODO

+ Add `"strict": true` to `.pyre_configuration`.

+ TODO: Pyre needs to show the `Literal` types. Right now, it weakens them to `int`, which means we don't see the exact values: `Revealed type for actual_prediction is tf.Tensor[tf.float32, int]`.

+ TODO: Pyre shouldn't weaken literals in some cases

	```
	outputs: List[Tensor[double, N1, L[1]]] = []
	foo: Tensor[double, N1, L[1]]
	outputs += [foo]
	reveal_type([foo])


	time_sequence_prediction/train.py:40:23 Incompatible parameter type [6]: Expected `typing.Iterable[Tensor[torch.float64, Variable[N1 (bound to int)], typing_extensions.Literal[1]]]` for 1st positional only parameter to call `list.__iadd__` but got `typing.Iterable[Tensor[torch.float64, Variable[N1 (bound to int)], int]]`.
	time_sequence_prediction/train.py:41:12 Revealed type [-1]: Revealed type for `[foo]` is `List[Tensor[torch.float64, Variable[N1 (bound to int)], int]]`.
	```

+ TODO: Pyre - Would be great if we could assign one method to another and preserve the type. Right now, this doesn't work:

	```
	def forward(self, x: Tensor[double, N1, N2]) -> Tensor[double, N1, N2]: ...
	__call__ = forward
	```

	It works out of the box in [Mypy](https://mypy-play.net/?mypy=latest&python=3.8&gist=08d857dc4403457f3a72ccdfbea6d0f0). Pyre does recognize the correct type but it doesn't automatically use it: `Attribute __call__ of class Sequence has type ... but no type is specified`. And I couldn't just write the `Callable` type because it has overloads and Python by itself doesn't have syntax for that (plus, it'd be really ugly). Maybe we can special-case `__call__`.

## Gotchas

+ We need to annotate empty list assignments: `x = []`. Otherwise, Pyre is unable to guess the eventual type. (We hope to change that in the future, but this is going to be a limitation for a while.)

+ We need to annotate attributes: `self.lstm1: nn.LSTMCell[L[1], L[51]] = nn.LSTMCell(1, 51)`. Pyre doesn't infer these automatically (because of reasons).

## Features we will need

+ Matrix multiplication requires a transpose operator.

+ Need some way of extracting a tuple from a list. For example, `tf.zeros([3, 4])` should return `Tensor[float32, L[3], L[4]]`. However, we have no way to extract individual types from a list. Note that we handle a tuple just fine: `tf.zeros((3, 4))`.

  One way is to accept a literal list and have the typechecker convert it to a tuple type.

  Another way is to have a `FiniteList[Unpack[Ts]]`, which has types `Ts` and behaves in all ways like a `Tuple`, except that it is initialized with a `[1, 2, 3]`.

+ Concatenation of multiple variadics

  - `argmax`

  - Will also need this for arbitrary indexing: `tensor.size(1)` gives the 2nd item in the shape tuple. We could type this using:

	```
	def size(self: Tensor[*Ts, T, *Rs], axis: Length[Ts]) -> T: ...
	```

	However, from my search of large codebases, almost all calls to `tensor.size(...)` used either axis=0, 1, 2, or 3. So, we can readily handle this with overloads.

+ Broadcasting - this is intended to be part of type arithmetic, but the relevant code has not yet landed in Pyre.

+ Better error messages. They are pretty arcane now.

+ Would be great to bind a type variable based on a default value. Otherwise, we have to add an overload for the default value.

	```
	@overload
	def zeros(
		shape: Tuple[Unpack[Ts]], dtype: Type[T], name: str = ...
	) -> Tensor[T, Unpack[Ts]]: ...
	@overload
	def zeros(
		shape: Tuple[Unpack[Ts]], dtype: Type[float32] = ..., name: str = ...
	) -> Tensor[float32, Unpack[Ts]]: ...


	# Ideally:
	def zeros(
		shape: Tuple[Unpack[Ts]], dtype: Type[T] = float32, name: str = ...
	) -> Tensor[T, Unpack[Ts]]: ...

	reveal_type(zeros(10, 20)) # => Tensor[float32, L[10], L[20]]
	reveal_type(zeros(10, 20), dtype=int) # => Tensor[int, L[10], L[20]]
	```

	Another example:

	```
	@overload
	def forward(
		self, input: Tensor[double, N1, N2], future: L[0] = 0
	) -> Tensor[double, N1, N2]: ...
	@overload
	def forward(
		self, input: Tensor[double, N1, N2], future: N3
	) -> Tensor[double, N1, Add[N2, N3]]: ...

	# Ideally:
	def forward(
		self, input: Tensor[double, N1, N2], future: N3 = 0
	) -> Tensor[double, N1, Add[N2, N3]]: ...
	```

	One blocker is that stubs are supposed to have ellipses instead of default values: `dtype: Type[T] = ...`. We'd have to change that.

+ Using a zero-dimensional Tensor as a float: I got an error when passing a function to `optimizer.step`

  > Expected Optional[Callable[[], float]] got Callable[[], Tensor[float64]].

  I'm guessing that Tensor with zero dimensions should be treated as a float.

## Notes on StackOverflow bugs

From https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow

+ UT-2 - tensorflow_program_bugs/ut_2_multiplication.py - this needs broadcasting (from type arithmetic).


+ UT-3 - tensorflow_program_bugs/ut_3_image_set_shape.py - I'm using `tf.ensure_shape` instead of `tf.set_shape`.

  Note that `x.set_shape` can't be typed statically since it changes the type parameters of a Tensor. Basically, `x: Tensor[L[10], L20]; x.set_shape([2, 10, 10])` will make the tensor have shape `2x10x10` at runtime. The Python type system has no way of reflecting the updated `self` type in the return signature. So, I'm using the [TensorFlow-recommended](https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape) `x.ensure_shape` instead, which returns a new Tensor. We can make the returned tensor type have the new shape.

+ UT-4 - this seems to be based on TensorFlow v1 style of `placeholder` code. That seems to be deprecated in v2. I narrowed it down to the parts that caused the shape errors.
