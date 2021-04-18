import numpy as np
from pyre_extensions import Add, Multiply, TypeVarTuple
from typing import Tuple, TypeVar
from typing_extensions import Literal as L
import tensorflow as tf
from tensorflow import float32, Tensor, Variable
import tensorflow_datasets as tfds


Dims = TypeVarTuple("Dims")


def weight_variable(shape: Tuple[*Dims]) -> Variable[float32, *Dims]:
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


Batch = TypeVar("Batch", bound=int)
Height = TypeVar("Height", bound=int)
Width = TypeVar("Width", bound=int)
FilterHeight = TypeVar("FilterHeight", bound=int)
FilterWidth = TypeVar("FilterWidth", bound=int)
InChannels = TypeVar("InChannels", bound=int)
OutChannels = TypeVar("OutChannels", bound=int)


def conv2d(
        x: Tensor[float32, Batch, Height, Width, InChannels],
        W: Tensor[float32, FilterHeight, FilterWidth, InChannels, OutChannels]
) -> Tensor[
        float32,
        # (Height - FilterHeight) / Stride + 1
        Add[Add[Height, Multiply[FilterHeight, L[-1]]], L[1]],
        # (Width - FilterWidth) / Stride + 1
        Add[Add[Width, Multiply[FilterWidth, L[-1]]], L[1]],
        OutChannels
]:
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable((5, 5, 1, 32))
b_conv1 = bias_variable((32,))
W_conv2 = weight_variable((5, 5, 32, 64))
b_conv2 = bias_variable((64,))
W_fc1 = weight_variable((7 * 7 * 64, 1024))
b_fc1 = bias_variable((1024,))
W_fc2 = weight_variable((1024, 10))
b_fc2 = bias_variable((10,))
parameters = (W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)


def compute_prediction(x: Tensor[float32, L[50], L[28], L[28], L[1]], keep_prob):
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, (-1, 7 * 7 * 64))
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, rate=(1-keep_prob))

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv


def compute_loss(x, y_, keep_prob):
    y_conv = compute_prediction(x, keep_prob)
    y_onehot = tf.one_hot(y_, depth=10)
    cross_entropy = -tf.reduce_sum(y_onehot * tf.math.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
    return cross_entropy


optimizer = tf.keras.optimizers.Adam(1e-4)


def train_step(x, y_, keep_prob):
    with tf.GradientTape() as tape:
        cross_entropy = compute_loss(x, y_, keep_prob)
    grads = tape.gradient(cross_entropy, parameters)
    optimizer.apply_gradients(zip(grads, parameters))


def compute_accuracy(x, y_, keep_prob):
    y_conv = compute_prediction(x, keep_prob)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def load_dataset(split):
    dataset = tfds.load(name="mnist", split=split)
    def normalize(example):
        example['image'] = tf.cast(example['image'], tf.float32) / 255.0
        return example
    dataset = dataset.map(normalize)
    return dataset


train_dataset = load_dataset(tfds.Split.TRAIN)
train_dataset = train_dataset.batch(50)
train_dataset = train_dataset.repeat()
train_dataset_iterator = iter(train_dataset)
for i in range(1000):
    batch = next(train_dataset_iterator)
    if i % 100 == 0:
        loss = compute_loss(x=batch['image'], y_=batch['label'], keep_prob=1.0)
        train_accuracy = compute_accuracy(x=batch['image'], y_=batch['label'], keep_prob=1.0)
        print("step %d, training accuracy %g, loss %g" % (i, train_accuracy, loss))
    train_step(x=batch['image'], y_=batch['label'], keep_prob=0.5)

test_dataset = load_dataset(tfds.Split.TEST)
test_tuples = ((x['image'], x['label']) for x in test_dataset)
test_images, test_labels = zip(*test_tuples)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
print("test accuracy %g" % compute_accuracy(x=test_images, y_=test_labels, keep_prob=1.0))
