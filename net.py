# -*- coding: utf-8 -*-

import os
import tensorflow as tf


class NeuralNetwork(object):

    def __init__(self, input_size, output_size, hidden_layer_sizes,
                 learning_rate, dtype=tf.float32):
        self._x = tf.placeholder(dtype, [None, input_size])
        self._weights = []
        self._biases = []
        self._layers = []

        prev_layer = self._x
        for layer_size in hidden_layer_sizes:
            # Add hidden layer with RELU activation.
            prev_layer = self._add_layer(prev_layer, input_size, layer_size,
                                         tf.nn.relu)
            input_size = layer_size

        # Add output layer with linear activation.
        self._y = self._add_layer(prev_layer, input_size, output_size,
                                  lambda x: x)

        # Set up trainer.
        self._y_truth = tf.placeholder(dtype, [None, output_size])
        loss = tf.reduce_mean(tf.square(self._y_truth - self._y)) / 2
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self._trainer = optimizer.minimize(loss)

        # Set up session.
        self._session = tf.Session()
        self._session.run(tf.initialize_all_variables())

        self._saver = tf.train.Saver()

    def _add_layer(self, prev_layer, input_size, output_size, activation):
        # Build layer.
        W = tf.Variable(tf.truncated_normal([input_size, output_size]))
        b = tf.Variable(tf.zeros([output_size]))
        out = activation(tf.matmul(prev_layer, W) + b)

        # Maintain references.
        self._weights.append(W)
        self._biases.append(b)
        self._layers.append(out)

        return out

    @property
    def weights(self):
        return self._session.run(self._weights)

    @property
    def biases(self):
        return self._session.run(self._biases)

    def evaluate(self, x):
        return self.evaluate_many([x])[0]

    def evaluate_many(self, xs):
        return self._session.run(self._y, feed_dict={self._x: xs})

    def train(self, x, y):
        self.train_many([x], [y])

    def train_many(self, xs, ys):
        self._session.run(self._trainer,
                          feed_dict={self._x: xs, self._y_truth: ys})

    def save(self, save_path="model.ckpt"):
        save_path = self._saver.save(self._session, save_path)
        print("model saved to file: {}".format(save_path))

    def load(self, load_path="model.ckpt"):
        if os.path.exists(load_path):
            self._session = tf.Session()
            self._saver.restore(self._session, load_path)
            print("model restored from: {}".format(load_path))


class Controller(object):

    def __init__(self, learning_rate=0.01):
        self._nn = NeuralNetwork(19, 3, [11, 7], learning_rate)

    def evaluate(self, x):
        return self._nn.evaluate(x)

    def train(self, x, actual):
        self._nn.train(x, actual)

    def load(self):
        self._nn.load()

    def save(self):
        self._nn.save()
