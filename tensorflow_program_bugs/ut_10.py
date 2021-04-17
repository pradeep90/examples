"""UT-10

Adapted to TensorFlow 2 from
https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow/UT-10/36343542-buggy/tflin.py
which, in turn, is originally from
https://stackoverflow.com/questions/36343542/tensorflow-shape-error-in-feed-dict

There are actually two bugs in this program.

The immediate bug is that while `compute_cost` can be called on a single
example without issue, when called on the whole of `train_X` and `train_Y`,
we get the following error at runtime:

    ValueError: Dimensions must be equal, but are 6 and 60000 for
    '{{node sub}} = Sub[T=DT_FLOAT](StatefulPartitionedCall, Y)'
    with input shapes: [60000,6], [60000].

This occurs while trying to compute the error between the prediction
and the true label, `activation - Y`. This works when `compute_cost`
is being called with only a single example, because there,
`X` has shape `(6,)`, so `activation` ends up also having shape `(6,)`,
and `Y` has shape `()`, so the operation succeeds through broadcasting.

But that brings us to the second bug: if each label is a scalar, then
`activation` should be a scalar too, not shape `(6,)`! It turns out
that `compute_activation` is broken too: it should be doing a dot product
between the weight `W` and the feature vector(s) `X`, but it's actually
doing point-wise multiplication. (As it happens, though, when the first bug
is fixed, training probably still succeeds, because the sum is done later
in `compute_cost` by `reduce_sum`.)

Can we detect the second bug 'for free' with shape-typing (that is, purely with
shape-stubs for TensorFlow, but without adding extra type annotations to this
code)? I don't think so; this is just a case where the user has accidentally
implemented the algorithm incorrectly.

Can we detect the first bug with typing? Unfortunately, I don't think so either:
in Pyre, at least, inside `compute_cost` we don't know the shape of `X` and `Y`
unless we annotate their expected types specifically.
"""

import sys
import tensorflow as tf
import numpy

# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50

# data:
# 1, 0, 0, 1, 1, 0, 0
# 1, 0, 0, 1, 1, 0, 0
# 0, 1, 0, 1, 1, 0, 0
# 0, 1, 0, 1, 0, 1, 0
# 1, 0, 1, 0, 0, 1, 0
# 1, 0, 1, 0, 0, 1, 0
# 1, 0, 0, 1, 1, 0, 0
# 1, 0, 0, 1, 1, 0, 0
# 1, 0, 0, 1, 1, 0, 0
# 1, 0, 1, 0, 0, 1, 0

f = open("ut_10_data.csv")
data = numpy.loadtxt(f, delimiter=",")
data = data.astype(numpy.float32)

train_X = data[:60000, :-1]
train_Y = data[:60000, -1]

test_X = data[60000:80000, :-1]
test_Y = data[60000:80000, -1]

X_val = data[80000:, :-1]
y_val = data[80000:, -1]

# Training Data
n_input = train_X.shape[1]
n_samples = train_X.shape[0]

print(n_input)

# Create Model

# Set model weights
W = tf.Variable(tf.zeros((6,)), name="weight")
b = tf.Variable(tf.zeros((1,)), name="bias")

optimizer = tf.keras.optimizers.SGD(learning_rate)

@tf.function
def compute_activation(X, Y):
    # Construct a linear model
    activation = tf.add(tf.multiply(X, W), b)
    return activation

@tf.function
def compute_cost(X, Y):
    activation = compute_activation(X, Y)
    # Minimize the squared errors
    cost = tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * n_samples)  # L2 loss
    return cost

@tf.function
def step(x, y):
    with tf.GradientTape() as tape:
        cost = compute_cost(x, y)
    grads = tape.gradient(cost, (W, b))
    optimizer.apply_gradients(zip(grads, (W, b)))

# Fit all training data
for epoch in range(training_epochs):
    for (x, y) in zip(train_X, train_Y):
        step(x, y)

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=",
              "{:.9f}".format(compute_cost(train_X, train_Y),
              "W=", W, "b=", b))

print("Optimization Finished!")
training_cost = compute_cost(train_X, train_Y)
print("Training cost=", training_cost, "W=", W, "b=", b, '\n')

print("Testing... (L2 loss Comparison)")
testing_cost = tf.reduce_sum(tf.pow(compute_activation(test_X, test_Y) - test_Y, 2)) / (2 * test_X.shape[0])
print("Testing cost=", testing_cost)
print("Absolute l2 loss difference:", abs(training_cost - testing_cost))
