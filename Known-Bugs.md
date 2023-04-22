
# Bugs in Pyre

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

+ Fix bug in type arithmetic: Bug in Pyre where `N + 1 - 1` is not treated as compatible with `N`.

```
from typing import Generic, TypeVar
from pyre_extensions import TypeVarTuple, Add
from typing_extensions import Literal as L

Height = TypeVar("Height", bound=int)

def foo(x: Height) -> Add[Add[Height, L[-1]], L[1]]: ...

def bar(x: Height) -> Height:
  return foo(x)

Incompatible return type [7]: Expected `Variable[Height (bound to int)]` but got `Variable[Height (bound to int)]`.

debug data:

120 2021-04-17 23:55:00 DUMP solve: start - Variable[test.Height (bound to int)] <: Variable[test.Height (bound to int)]
    121 (IntExpression
    122  (((constant_factor 1)
    123    (variables
    124     (((variable
    125        (Variable
    126         ((variable test.Height) (constraints (Bound (Primitive int)))
    127          (variance Invariant) (state InFunction) (namespace 0))))
    128       (degree 1))))))) <: (Variable
    129  ((variable test.Height) (constraints (Bound (Primitive int)))
    130   (variance Invariant) (state InFunction) (namespace 0)));
    131 constraints: {
    132 Have Fallbacks to Any: ()}
```

+ Plain `Tensor` is not handled gracefully:

```
def foo(x: Tensor) -> Tensor: ...

def bar(x: Tensor[int, L[10], L[20]]) -> None: ...

def baz(x: Tensor, y: Tensor[int, L[10], L[20]]) -> None:
    foo(x)
    foo(y)

	# Error here.
    bar(x)
    bar(y)

$ pyre
Incompatible parameter type [6]: Expected `Tensor[int, typing_extensions.Literal[10], typing_extensions.Literal[20]]` for 1st positional only parameter to call `bar` but got `Tensor[typing.Any, *Tuple[typing.Any, ...]]`.
```

This basically needs `Tuple[Any, ...]` to be compatible with `Tuple[L[10], L[20]]`. Will have to add that.

# Gotchas

+ We need to annotate empty list assignments: `x = []`. Otherwise, Pyre is unable to guess the eventual type. (We hope to change that in the future, but this is going to be a limitation for a while.)

+ We need to annotate attributes: `self.lstm1: nn.LSTMCell[L[1], L[51]] = nn.LSTMCell(1, 51)`. Pyre doesn't infer these automatically (because of reasons).

# Features we will need

+ Need some way of extracting a tuple from a list. For example, `tf.zeros([3, 4])` should return `Tensor[float32, L[3], L[4]]`. However, we have no way to extract individual types from a list. Note that we handle a tuple just fine: `tf.zeros((3, 4))`.

  One way is to accept a literal list and have the typechecker convert it to a tuple type.

  Another way is to have a `FiniteList[Unpack[Ts]]`, which has types `Ts` and behaves in all ways like a `Tuple`, except that it is initialized with a `[1, 2, 3]`.

+ Similarly, we need a way of inferring the shape of nested Python lists. For example, if `xs = [[0 for _ in range(10)]]`, we should be able to infer it has shape `(1, 10)`.

+ Concatenation of multiple variadics

  - `argmax`

  - Will also need this for arbitrary indexing: `tensor.size(1)` gives the 2nd item in the shape tuple. We could type this using:

	```
	def size(self: Tensor[*Ts, T, *Rs], axis: Length[Ts]) -> T: ...
	```

	However, from my search of large codebases, almost all calls to `tensor.size(...)` used either axis=0, 1, 2, or 3. So, we can readily handle this with overloads.

+ Broadcasting - this is intended to be part of type arithmetic, but the relevant code has not yet landed in Pyre.

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

+ Infer literal types after basic arithmetic

  example: we are unable to infer that `reflection_padding` has type `Divide[KernelSize, 2]`. Hence the `ignore` on the next line:

  ```python
	reflection_padding = kernel_size // 2
	self.reflection_pad: torch.nn.ReflectionPad2d[Divide[KernelSize, L[2]]] = torch.nn.ReflectionPad2d(reflection_padding)  # type: ignore
  ```

+ Succinct type arithmetic syntax: Right now, `black` splits a big return type across a dozen lines.

+ Would be great to have `Subtract[N, M]` instead of `Add[N, Multiply[M, L[-1]]]`.

## Notes on StackOverflow bugs

From https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow

+ UT-2 - tensorflow_program_bugs/ut_2_multiplication.py - this needs broadcasting (from type arithmetic).


+ UT-3 - tensorflow_program_bugs/ut_3_image_set_shape.py - I'm using `tf.ensure_shape` instead of `tf.set_shape`.

  Note that `x.set_shape` can't be typed statically since it changes the type parameters of a Tensor. Basically, `x: Tensor[L[10], L20]; x.set_shape([2, 10, 10])` will make the tensor have shape `2x10x10` at runtime. The Python type system has no way of reflecting the updated `self` type in the return signature. So, I'm using the [TensorFlow-recommended](https://www.tensorflow.org/api_docs/python/tf/Tensor#set_shape) `x.ensure_shape` instead, which returns a new Tensor. We can make the returned tensor type have the new shape.

+ UT-4 - this seems to be based on TensorFlow v1 style of `placeholder` code. That seems to be deprecated in v2. I narrowed it down to the parts that caused the shape errors.

### UT-11

This is purely a tensor-placeholder size mismatch bug, which probably isn't that interesting to cover since so few folks use TensorFlow 1 these days.

### UT-12

The actual issue here is that the programmer tried to flatten a tensor using `reshape(h, [-1, 4 * 4 * 64])`. Unfortunately, they got the size wrong - the tensor was actually shape `batch x 8 x 8 x 64`. This doesn't actually cause an error until later in the code, when the incorrect flattening causes a later `reshape` to result in a tensor with the wrong batch size.

While we _could_ statically detect the same error as is produced at runtime, I'm not sure that would be very interesting - that error doesn't help in narrowing down what the problem really is. And I don't think we can detect the problem with the reshape with static analysis, because the programmer simply told the program to do the wrong thing.

The kind of thing that would _actually_ help the programmer here would either be:

* A system which shows statically-inferred shapes to the user _while_ they're coding.
* Type annotations specifying what the user expects the shape after each layer to be, which could be checked _after_ the user has finished coding.

Unfortunately, both of these involve complicated calculations to determine how convolutions and pooling change the shape, which we can't do without type arithmetic.

So yeah, however we slice this one, I don't think we can do anything useful :(

### UT13

The problem here is caused by a comparison to `argmax(Y)` instead of `Y` (a scalar) itself.

I don't think we can catch this with static shape analysis. It doesn't produce a runtime error; it's valid code, but it just does the wrong thing. I think this code has been copy-pasted from a different example without the programmer really understanding what was going on.

# Tips for Future Stub Authors

+ Consider writing a "unit test" function for complicated type signatures.

If we are doing non-trivial arithmetic in a type signature, it helps to have tests that can tell us that we did it right and can catch regressions later on.

For example:

```
class ConvLayer(Module, Generic[InChannels, OutChannels, KernelSize, Stride]):
    def __call__(
            self,
            x: Tensor[DType, Batch, InChannels, Height, Width]
    ) -> Tensor[
        DType,
        Batch,
        OutChannels,
        Add[Divide[Add[Add[Height, Multiply[KernelSize, L[-1]]], Multiply[Divide[KernelSize, L[2]], L[2]]], Stride], L[1]],
        Add[Divide[Add[Add[Width, Multiply[KernelSize, L[-1]]], Multiply[Divide[KernelSize, L[2]], L[2]]], Stride], L[1]],
    ]: ...

def test_conv_layer_type() -> None:
    conv: ConvLayer[L[10], L[20], L[3], L[1]]
    x: Tensor[int, L[2], L[10], L[3], L[5]]
	y: Tensor[int, L[2], L[20], L[3], L[5]] = conv(x)
```

# Need for Better Error Messages

They are pretty arcane now.

+ Doesn't show the existing literal types bound. It shows `Channel` instead of `L[3]` for the parameter `InChannels` of `__call__`.

```
class ConvLayer(Module, Generic[InChannels, OutChannels, KernelSize, Stride]):
    def __call__(
            self,
            x: Tensor[DType, Batch, InChannels, Height, Width]
    ) -> Tensor[
        DType,
        Batch,
        OutChannels,
        Add[Divide[Add[Add[Height, Multiply[KernelSize, L[-1]]], Multiply[Divide[KernelSize, L[2]], L[2]]], Stride], L[1]],
        Add[Divide[Add[Add[Width, Multiply[KernelSize, L[-1]]], Multiply[Divide[KernelSize, L[2]], L[2]]], Stride], L[1]],
    ]: ...

conv1: ConvLayer[Channels, Channels, L[3], L[1]] = ConvLayer(channels, channels, kernel_size=3, stride=1)

x: Tensor[int, L[2], L[10], L[3], L[5]]
conv1(x)

fast_neural_style/neural_style/transformer_net.py:132:19 Incompatible parameter type [6]: Expected `Tensor[Variable[DType], Variable[Batch (bound to int)], Variable[Channels (bound to int)], Variable[Height (bound to int)], Variable[Width (bound to int)]]` for 1st positional only parameter to call `ConvLayer.__call__` but got `Tensor[int, int, int, int, int]`.
```

+ Truly baffling: Didn't realize that I was using `2` instead of `L[2]`.

```
y: Divide[L[3], L[2]]

fast_neural_style/neural_style/transformer_net.py:103:11 Invalid type [31]: Expression `pyre_extensions.Divide[(typing_extensions.Literal[4], 2)]` is not a valid type.
fast_neural_style/neural_style/transformer_net.py:103:11 Invalid type parameters [24]: Type parameter `unknown` violates constraints on `Variable[pyre_extensions._B (bound to int)]` in generic type `Divide`.
```

+ Don't see `reveal_type` in .pyi files. Confusing.

+ `Divide[KernelSize, 2]` shows up as `IntExpression`, which may seem like it's not something that can be assigned to a `TypeVar` with `bound=int`. We should clarify that it is indeed assignable.

+ Used `TypeVar("Padding")` instead of `TypeVar("Padding", bound=int)`. The revealed type for `Add[Add[Height, Padding], Padding]` was `Any`, without no explanation. Should point out that we used a non-int variable.

+ Revealed type for a callable should instantiate known literal types. That way, users can tell what the expected type is. Below, it should have shown an expected parameter type of `Tensor[DType, Batch, Channels, Height, Width]` and a return type of `Tensor[DType, Batch, Channels, ..., ...]`.

```
class ConvLayer(Module, Generic[InChannels, OutChannels, KernelSize, Stride]):
    def __call__(
            self,
            x: Tensor[DType, Batch, InChannels, Height, Width]
    ) -> Tensor[
        DType,
        Batch,
        OutChannels,
        Add[Divide[Add[Add[Height, Multiply[KernelSize, L[-1]]], Multiply[Divide[KernelSize, L[2]], L[2]]], Stride], L[1]],
        Add[Divide[Add[Add[Width, Multiply[KernelSize, L[-1]]], Multiply[Divide[KernelSize, L[2]], L[2]]], Stride], L[1]],
    ]: ...

conv1: ConvLayer[Channels, Channels, L[3], L[1]] = ConvLayer(channels, channels, kernel_size=3, stride=1)

fast_neural_style/neural_style/transformer_net.py:133:8 Revealed type [-1]: Revealed type for `self.conv1` is `ConvLayer[Variable[Channels (bound to int)], Variable[Channels (bound to int)], typing_extensions.Literal[3], typing_extensions.Literal[1]]`.

fast_neural_style/neural_style/transformer_net.py:133:8 Revealed type [-1]: Revealed type for `self.conv1.__call__` is `BoundMethod[typing.Callable(ConvLayer.__call__)[[Named(self, ConvLayer[Variable[Channels (bound to int)], Variable[Channels (bound to int)], typing_extensions.Literal[3], typing_extensions.Literal[1]]), Named(x, Tensor[Variable[DType], Variable[Batch (bound to int)], Variable[Channels (bound to int)], Variable[Height (bound to int)], Variable[Width (bound to int)]])], Tensor[Variable[DType], Variable[Batch (bound to int)], Variable[Channels (bound to int)], Variable[Height (bound to int)], Variable[Width (bound to int)]]], ConvLayer[Variable[Channels (bound to int)], Variable[Channels (bound to int)], typing_extensions.Literal[3], typing_extensions.Literal[1]]]`.
```

+ Should instantiate literal types:

```
fast_neural_style/neural_style/transformer_net.py:148:20 Unsupported operand [58]: `+` is not supported for operand types `Tensor[Variable[DType], Variable[Batch (bound to int)], Variable[Channels (bound to int)], Variable[Height (bound to int)], Variable[Width (bound to int)]]` and `Tensor[Variable[DType], Variable[Batch (bound to int)], Variable[Channels (bound to int)], Variable[Height (bound to int)], Variable[Width (bound to int)]]`.
```
