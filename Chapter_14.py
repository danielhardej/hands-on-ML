# ------------------------------------------------------------------------------
#                   Chapter 14: Recurrent Neural Networks
# ------------------------------------------------------------------------------
from __future__ import division, print_function, unicode_literals
import numpy as np
import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# -------------------- A basic RNN without TensorFlow RNN ops ------------------
n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf.global_variables_initializer()

X0_batch = np.array([[0,1,2], [3,4,5], [6,7,8], [9,0,1]])
X1_batch = np.array([[9,8,7], [0,0,0], [6,5,4], [3,2,1]])

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

print(Y0_val)
print(Y1_val)

# ------------------------ Static Unrolling Through Time -----------------------
"""The static_rnn() function creates an unrolled RNN by chaining cells."""
# X0 = tf.placeholder(tf.float32, [None, n_inputs])
# X1 = tf.placeholder(tf.float32, [None, n_inputs])
#
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype=tf.float32)
#
# Y0, Y1 = output_seqs

# But for lots of times-steps, this method would not be very good - we'd have to
# define an input and outp tensor for each step.
n_steps = 2
#
# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# X_seqs = tf.unstack(tf.transpose(X, perm=[1,0,2]))
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
# outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])
#
# X_batch = np.array([
#         [[0, 1, 2], [9, 8, 7]],
#         [[3, 4, 5], [0, 0, 0]],
#         [[6, 7, 8], [6, 5, 4]],
#         [[9, 0, 1], [3, 2, 1]],
#     ])
#
# init = tf.global_variables_initializer()
# # init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     init.run()
#     outputs_val = outputs.eval(feed_dict={X: X_batch})
#
# print(outputs_val)
# But the above approach still builds a graph for each cell for each time step.
# For a lot of time steps (say, 50+), you could get OOM errors. Fortunately,
# there is a solution: the dynamic_rnn() funtion!

# ------------------------ Dynamic Unrolling Through Time ----------------------
# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# -------------------------------- Training RNNs -------------------------------
# Training a sequence classifier
n_steps = 28
# n_inputs = 28
# n_neurons = 150
# n_outputs = 10
#
# learning_rate = 0.001
#
# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# y = tf.placeholder(tf.int32, [None])
#
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
#
# logits = tf.layers.dense(states, n_outputs)
# xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#
# loss = tf.reduce_mean(xentropy)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(loss)
# correct = tf.nn.in_top_k(logits, y, 1)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# init = tf.global_variables_initializer()
#
# mnist = input_data.read_data_sets("/tmp/data/")
#
# X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
# y_test = mnist.test.labels
#
# n_epochs = 100
# batch_size = 150
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for iteration in range(mnist.train.num_examples // batch_size):
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             X_batch = X_batch.reshape((-1, n_steps, n_inputs))
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#         print("Epoch:", epoch, "Train accuracy:", acc_train*100 , "Test accuracy:", acc_test*100)

# Training to predict time series
t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))
n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_iterations = 1500
batch_size = 50

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    saver.save(sess, "./my_time_series_model")

with tf.Session() as sess:                                                      # make a prediction using the model
    saver.restore(sess, "./my_time_series_model")

    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})

print(y_pred)
