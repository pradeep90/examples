import tensorflow as tf
import numpy as np
import random

num_features = 10
num_classes = 2
num_data_points = 500
xs = np.random.normal(0, 0.1, (num_data_points, num_features)).astype("float32")
# We can't infer the shape of this version :(
# ys = [[random.randint(0, num_classes) for _ in range(num_data_points)]]
# Instead:
ys = np.random.randint(0, num_classes + 1, (1, num_data_points))

w = tf.Variable(tf.zeros((num_features, num_classes)))
b = tf.Variable(tf.zeros((num_classes,)))

optimizer = tf.keras.optimizers.SGD(1e-4)

for _ in range(100):
    with tf.GradientTape() as tape:
        logits = tf.matmul(xs, w) + b

        # Broken:
        # predictions = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ys)
        #
        # TensorFlow says:
        #  InvalidArgumentError: logits and labels must be broadcastable:
        #  logits_size=[500,2] labels_size=[1,500]
        #  [Op:SoftmaxCrossEntropyWithLogits]
        #
        # Pyre says:
        #  Incompatible parameter type [6]: Expected
        #  `tf.Tensor[Variable[tf.nn.Batch (bound to int)], Variable[tf.nn.Features (bound to int)]]`
        #  for 1st parameter `logits` to call `tf.nn.softmax_cross_entropy_with_logits` but got
        # `tf.Tensor[int, int]`.

        # Possible correction 1:
        ys_onehot = tf.one_hot(ys[0], num_classes)
        predictions = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=ys_onehot
        )

        # Possible correction 2:
        ys_sparse = ys[0]
        predictions = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=ys_sparse
        )

        cost = tf.reduce_mean(predictions)
    grads = tape.gradient(cost, (w, b))
    optimizer.apply_gradients(zip(grads, (w, b)))
    print("Cost: {:.3f}".format(cost.numpy()))

# Note: this still doesn't train successfully.
# We can't fit random labels for 500 data points
# with linear regression.
