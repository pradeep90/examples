"""UT-12

Adapted to TensorFlow 2 from
https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow/UT-12/43285733-buggy/mnist.py
which, in turn, is originally from
https://stackoverflow.com/questions/43285733/incompatible-shapes-on-tensorflow
"""

from pyre_extensions import TypeVarTuple, Unpack
from typing import Tuple, TypeVar
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

Shape = TypeVarTuple("Shape")

n_data = 100
X = [np.random.uniform(0, 255, 900) for _ in range(n_data)]
X = np.asarray(X, dtype=np.float32)
y = []
for _ in range(n_data):
    y.append(np.zeros([62]))
    y[-1][random.randint(0, 61)] = 1

# normalise the features
X = (X - 255 / 2) / 255

y = np.float32(y)
X = np.float32(X)
Xtr, Xte, Ytr, Yte = train_test_split(X, y, train_size=0.7)

batch_size = 10

def weight_variable(shape: Tuple[Unpack[Shape]]) -> tf.Variable[tf.float32, Unpack[Shape]]:
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape: Tuple[Unpack[Shape]]) -> tf.Variable[tf.float32, Unpack[Shape]]:
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable((5, 5, 1, 32))
b_conv1 = bias_variable((32,))

W_fc1 = weight_variable((4 * 4 * 64, 1024))
b_fc1 = bias_variable((1024,))

W_conv2 = weight_variable((5, 5, 32, 64))
b_conv2 = bias_variable((64,))

W_fc2 = weight_variable((1024, 62))
b_fc2 = bias_variable((62,))

parameters = (W_conv1, b_conv1, W_fc1, b_fc1, W_conv2, b_conv2, W_fc2, b_fc2)

optimizer = tf.keras.optimizers.Adam(1e-4)

Batch = TypeVar("Batch")
Classes = TypeVar("Classes")
Features = TypeVar("Features")

@tf.function
def compute_logits(
        x: tf.Tensor[tf.float32, Batch, Features],
        y_: tf.Tensor[tf.float32, Batch, Classes],
        keep_prob: int
):
    x_image = tf.reshape(x, (-1, 30, 30, 1))

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, (-1, 4 * 4 * 64))
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, (1 - keep_prob))

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv

@tf.function
def train_step(x, y_, keep_prob):
    with tf.GradientTape() as tape:
        logits = compute_logits(x, y_)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    grads = tape.gradient(cross_entropy, parameters)
    optimiser.apply_gradients(zip(grads, parameters))

@tf.function
def compute_accuracy(x, y_, keep_prob):
    logits = compute_logits(x, y_, keep_prob)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

for i in range(20000):
    offset = (i * batch_size) % (Ytr.shape[0] - batch_size)
    batch_x = Xtr[offset:(offset + batch_size), :]
    batch_y = Ytr[offset:(offset + batch_size), :]
    if i % 100 == 0:
        train_accuracy = compute_accuracy(batch_x, batch_y, keep_prob=1.0)
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step(Xtr[offset:(offset + batch_size), :],
               Ytr[offset:(offset + batch_size), :],
               keep_prob=0.5)

print("test accuracy %g" % compute_accuracy(Xte, Yte, keep_prob=1.0))
