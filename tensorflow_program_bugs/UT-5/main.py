import os
import sys
import json
import time
import tensorflow as tf
import tensorflow_datasets as tfds


n_input = 28  # MNIST data input (img shape: 28*28)
n_steps = 28  # timesteps
n_hidden = 128  # hidden layer num of features
n_classes = 10  # MNIST total classes (0-9 digits)


# coding:utf-8
def rnn_model(x, lstm, weights, biases):
    """RNN (LSTM or GRU) model for image"""
    output = lstm(x)
    return tf.matmul(output, weights) + biases


def compute_loss(x, y, lstm, weights, biases):
    pred = rnn_model(x, lstm, weights, biases)
    # Define loss and optimizer
    # you will get the dreaded 'No gradients provided for any variable' if you switch the args between y and pred
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    return cost


def train_step(optimizer, x, y, lstm, weights, biases):
    with tf.GradientTape() as tape:
        cost = compute_loss(x, y, lstm, weights, biases)
    parameters = [weights, biases] + lstm.variables
    grads = tape.gradient(cost, parameters)
    optimizer.apply_gradients(zip(grads, parameters))


def compute_accuracy(x, y, lstm, weights, biases):
    pred = rnn_model(x, lstm, weights, biases)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def get_dataset(split):
    dataset = tfds.load("mnist", split=split)
    def normalize(x):
        x['image'] = tf.cast(x['image'], tf.float32) / 255.0
        return x
    dataset = dataset.map(normalize)
    return dataset


def train():
    """Train an image classifier"""
    """Step 0: load image data and training parameters"""
    # parameter_file = sys.argv[1]
    # params = json.loads(open(parameter_file).read())
    params = json.loads(open('parameters.json').read())

    train_dataset = get_dataset(tfds.Split.TRAIN)
    train_dataset = train_dataset.batch(params['batch_size'])
    train_dataset_iterator = iter(train_dataset)

    """Step 1: build a rnn model for image"""

    lstm = tf.keras.layers.LSTM(n_hidden)
    weights = tf.Variable(tf.random.normal([n_hidden, n_classes]), name='weights')
    biases = tf.Variable(tf.random.normal([n_classes]), name='biases')

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])


    """Step 2: train the image classification model"""
    step = tf.Variable(1)

    """Step 2.0: create a directory for saving model files"""
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckpt = tf.train.Checkpoint(step=step, lstm=lstm, weights=weights, biases=biases, optimizer=optimizer)

    """Step 2.1: train the image classifier batch by batch"""
    while step * params['batch_size'] < params['training_iters']:
        batch = next(train_dataset_iterator)
        batch_x = batch['image']
        batch_y = batch['label']
        # Reshape data to get 28 seq of 28 elements
        batch_x = tf.reshape(batch_x, (params['batch_size'], n_steps, n_input))
        # batch_y = batch_y.reshape((-1, n_classes))

        train_step(optimizer, batch_x, batch_y, lstm, weights, biases)

        """Step 2.2: save the model"""
        if step % params['display_step'] == 0:
            path = ckpt.save(checkpoint_prefix)
            acc = compute_accuracy(batch_x, batch_y, lstm, weights, biases)
            loss = compute_loss(batch_x, batch_y, lstm, weights, biases)
            print('Iter: {}, Loss: {:.6f}, Accuracy: {:.6f}'.format(step * params['batch_size'], loss, acc))
        step.assign_add(1)
    print("The training is done")

    """Step 3: test the model"""
    test_len = 128
    test_dataset = get_dataset(tfds.Split.TEST)
    test_dataset = test_dataset.batch(test_len)
    test_batch = next(iter(test_dataset))
    test_images = tf.reshape(test_batch['image'], (-1, n_steps, n_input))
    test_labels = test_batch['label']
    print("Testing Accuracy:", compute_accuracy(test_images, test_labels, lstm, weights, biases).numpy())


if __name__ == '__main__':
    train()
