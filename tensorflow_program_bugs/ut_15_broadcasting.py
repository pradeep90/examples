"""UT-15

Adapted to TensorFlow 2 from
https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow/UT-15/38447935-buggy/fitting.py
which, in, is originally from
https://stackoverflow.com/questions/38447935/tensorflow-model-always-produces-mean

The bug here occurs because `y_` has shape (3,), while `y` has shape (3, 1).
In `compute_error`, therefore, `y_ - y` ends up being broadcast to (3, 3),
which produces completely the wrong gradients.

I'm not sure we can detect this automatically with shape-typing.
"""

import tensorflow as tf


def weight_variable(shape):
    """Initialize the weights with random weights"""
    initial = tf.random.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)


# Initialize my data
x = tf.constant([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]], dtype=tf.float64)
y_ = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
w = weight_variable((2, 1))

def compute_predictions():
    y = tf.matmul(x, w)
    return y

def compute_error():
    y = compute_predictions()
    error = tf.reduce_mean(tf.square(y_ - y))
    return error

optimizer = tf.keras.optimizers.Adam(1e-4)

def train_step():
    with tf.GradientTape() as tape:
        error = compute_error()
    grads = tape.gradient(error, (w,))
    optimizer.apply_gradients(zip(grads, (w,)))

# Train the model and output every 1000 iterations
for i in range(50000):
    train_step()
    err = compute_error()

    if i % 1000 == 0:
        print("\nerr:", err)
        print("x: ", x)
        print("w: ", w)
        print("y_: ", y_)
        print("y: ", compute_predictions())
