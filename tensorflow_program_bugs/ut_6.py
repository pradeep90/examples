import tensorflow as tf
import numpy as np


n_input = 300
n_classes = 10
vocab_size = 100
embedding_size = 128
o1 = 64
o2 = 32
weights = {"out": tf.random.uniform((o2, 1)), "h1": tf.random.uniform((embedding_size, o1)),
           "h2": tf.random.uniform((o1, o2))}
biases = {"out": tf.random.uniform((1,)), "b1": tf.random.uniform((o1,)),
          "b2": tf.random.uniform((o2,))}

with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(tf.random.uniform((vocab_size, embedding_size), -1.0, 1.0), name="W")


def multilayer_perceptron(_X, _weights, _biases):
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        embedding_layer = tf.nn.embedding_lookup(W, _X)
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(embedding_layer, _weights['h1']), _biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    return tf.matmul(layer_2, weights['out']) + biases['out']


def compute_cost(x, y):
    pred = multilayer_perceptron(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    return cost


print(compute_cost(x=np.random.randint(0, 10, (2, n_input)),
                   y=[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]))
