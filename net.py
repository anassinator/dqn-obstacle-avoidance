# -*- coding: utf-8 -*-

import tensorflow as tf


class NeuralNetwork(object):

    def __init__(self, input_size, output_size, hidden_layer_sizes,
                 dtype=tf.float32):
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
        with tf.name_scope("error"):
            loss = tf.reduce_mean(tf.square(self._y_truth - self._y)) / 2
            tf.scalar_summary("error", loss)
            self.variable_summaries(loss)
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        self._trainer = optimizer.minimize(loss)

        # Set up session.
        self._session = tf.Session()

        # Set up summaries.
        self._summaries = tf.merge_all_summaries()
        self._train_writer = tf.train.SummaryWriter("logs/train",
                                              self._session.graph)
        self._test_writer = tf.train.SummaryWriter("logs/test",
                                                   self._session.graph)
        self._session.run(tf.initialize_all_variables())

        # Set up saver.
        self._saver = tf.train.Saver()

    def variable_summaries(self, var):
        mean = tf.reduce_mean(var)
        tf.scalar_summary("mean", mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary("stddev", stddev)
        tf.scalar_summary("max", tf.reduce_max(var))
        tf.scalar_summary("min", tf.reduce_min(var))
        tf.histogram_summary("histogram", var)

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
        summ, _ = self._session.run([self._summaries, self._trainer],
                                    feed_dict={self._x: xs, self._y_truth: ys})
        self._train_writer.add_summary(summ)

    def save(self, save_path="model.ckpt"):
        save_path = self._saver.save(self._session, save_path)
        print("Model saved in file: %s" % save_path)

    def load(self, save_path="model.ckpt"):
        self._session = tf.Session()
        self._saver.restore(self._session, save_path)
        print("Model restored.")


class Controller(object):

    def __init__(self):
        self._nn = NeuralNetwork(19, 7, [13])

    def evaluate(self, x):
        return self._nn.evaluate(x)

    def train(self, x, actual):
        self._nn.train(x, actual)
