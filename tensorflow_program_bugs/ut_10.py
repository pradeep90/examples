import sys
import tensorflow as tf
import numpy

rng = numpy.random

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
W = tf.Variable(tf.zeros([6]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

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
