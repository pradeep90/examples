"""UT-13

Adapted to TensorFlow 2 from
https://github.com/ForeverZyh/TensorFlow-Program-Bugs/blob/master/StackOverflow/UT-13/42191656-buggy/linear.py
which, in turn, is originally from
https://stackoverflow.com/questions/42191656/tensorflow-learning-xor-with-linear-function-even-though-it-shouldnt

The bug here lies in `compute_accuracy`, where we try to compute whether the
prediction is correct by comparing the argmax of the predicted vector to
the argmax of the ground-truth vector. Unfortunately, both the predicted
vector and the ground-truth vector have a length of 1, so this always succeeds.

I don't think we can catch the error here with type checking.
"""

import tensorflow as tf

############################################################
'''
    dummy data
'''
x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
y_data = [[0], [1], [1], [0]]

############################################################
'''
    Network parameters
'''
W = tf.Variable(tf.random.uniform((2, 2), -1, 1), name='W')
c = tf.Variable(tf.zeros((2,)), name='c')
w = tf.Variable(tf.random.uniform((2, 1), -1, 1), name='w')
b = tf.Variable(tf.zeros((1,)), name='b')

############################################################
'''
    Network 1:

    function: Yhat = (w (x'W + c) + b)
    loss    : \sum_i Y * log Yhat
'''

def compute_prediction(X):
    H1 = tf.matmul(X, W) + c
    Yhat1 = tf.matmul(H1, w) + b
    return Yhat1

optimizer = tf.keras.optimizers.Adam(0.01)

def train_step(X, Y):
    with tf.GradientTape() as tape:
        Yhat1 = compute_prediction(X)
        cross_entropy1 = tf.reduce_mean(tf.square(Y - Yhat1))
    grads = tape.gradient(cross_entropy1, (W, c, w, b))
    optimizer.apply_gradients(zip(grads, (W, c, w, b)))
    return Yhat1, cross_entropy1

'''
    Train
'''

for i in range(100):
    yhat, loss = train_step(x_data, y_data)
    print("loss %g" % loss)
    print(yhat)

'''
    Evaluation
'''

def compute_accuracy(X, Y):
    Yhat1 = compute_prediction(X)
    corrects = tf.equal(tf.argmax(Y, 1), tf.argmax(Yhat1, 1))
    accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
    return accuracy

r = compute_accuracy(x_data, y_data)
print('accuracy: ' + str(r * 100) + '%')
