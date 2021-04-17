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
W = tf.Variable(tf.random.uniform([2, 2], -1, 1), name='W')
c = tf.Variable(tf.zeros([2]), name='c')
w = tf.Variable(tf.random.uniform([2, 1], -1, 1), name='w')
b = tf.Variable(tf.zeros([1]), name='b')

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
