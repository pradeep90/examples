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
