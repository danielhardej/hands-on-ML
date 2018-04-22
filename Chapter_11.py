from __future__ import division, print_function, unicode_literals               # Needed for interations/loops to work properly!!!
import os
import tensorflow as tf
import numpy as np

# ----------- Part 1: Dealing with vanishing/exploding gradients ---------------
""" This code lists several functions for modifying activation functions of
neurons in a deep net to deal with exploding/vanishing gradients."""

# A regular ReLU with He initialization:
he_init = tf.contrib.layers.variance_scaling_initializer()
hidden_1 = tf.layer.dense(X, n_hidden_1, activation=tf.nn.relu, kernel_initializer=he_init, name="hidden_1")

# An exponential linear unit:
hidden_1 = tf.layers.dense(X, n_hidden_1, activation=tf.nn.elu, name="hidden_1")

# A 'leaky' ReLU (no predefined funciton in TensorFlow, but it's easy to make one):
def leaky_relu(z, name=None):
    return tf.maximum(0.01*z, z, name=name)

hidden_1 = tf.layers.dense(X, n_hidden_1, activation=leaky_relu, name="hidden_1")

# Using Batch Normalization to eliminate vanishing/exploding gradients
n_inputs = 28*28
n_hidden_1 = 300
n_hidden_2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

training = tf.placeholder_with_default(False, shape=(), name="training")
# NB no activation function has been specified!
hidden_1 = tf.layers.dense(X, n_hidden_1, name="hidden_1")                      # Create hidden layer 1
bn1 = tf.layer.batch_normalization(hidden_1, training=training, momentum=0.9)   # Batch normalization on first layer
bn1_act = tf.nn.elu(bn1)                                                        # Batch normalization passes through an ELU

hidden_2 = tf.layers.dense(bn1_act, n_hidden_2, name="hidden_2")                # Create hidden layer 2
bn2 = tf.layer.batch_normalization(hidden_2, training=training, momentum=0.9)   # Batch normalization on second layer
bn2_act = tf.nn.elu(bn2)                                                        # Batch normalization passes through an ELU

logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")          # Generate output layer
logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)       # Batch normalization on output layer

# The rest of the construction phase is as normal: define the cost funcion,
# create an optimizer, tell the optimizer to minimize the cost function, define
# the evaluation operations, create a Saver etc.

# The execution phase is slightly different though. Most steps are the same, but
# there are a few extra steps...
# 1) when running an operation using batch_normalization during training, you
#    you need to set the training placeholder =True
# 2) batch_normalization creates function that must be evaluated at each step
#    during training to update moving averages. These operations are added to the
#    UPDATE_OPS collection, so we just need to get the list of operations and run
#    them at each training iteration.
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print("Epoch: ", epoch, "Test accuracy: ", accuracy_val)
    save_path = saver.save(sess, "./my_model_final.ckpt")

# Gradient clipping - clipping gradients during backprop so that they never
# exceed some threshold:
threshold = 1.0                                                                 # NB threshold hyperparameter can be tuned!
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)                              # Computes only the gradients without applying them
clipped_grads = [tf.clip_by_value(grad, threshold, -threshold),                 # Clips gradients computed in previous code
                    for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(clipped_grads)
# NB above code is run at every training step!


# -------------- Part 2: Speeding-up training of large models ------------------
# Reusing a TensorFlow model:
saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")                #

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")
accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")
training_op = tf.get_default_graph().get_tensor_by_name("GradientDescent")

for op in tf.get_default_graph().get_operations():
    print(op.name)

# Freezing lower layers:
train_vars = tf.get_collection(tf.GraphKeys.TRAINING_VARIABLES, scope="hidden[34]|outputs")     # Gets list of all trainable variables in layers 3 and 4 and output
training_op = optimizer.minimize(loss, var_list=train_vars)                     # Provide list of trainable variables to the optimizer
# Caching frozen layers:                                                        # NB Go to the Hands-On ML GitHub for the complete code!
n_batches = mnist.train.num_examples // batch_size

with tf.Session() as sess:                                                      # The following code runs the whole training set through the lower frozen
    init.run()                                                                  # layers of the network. So during training, batches of outputs from the highest
    restore_saver.restore(sess, "./my_model_final.ckpt")                        # frozen layer are fed to the training operation.

    h2_cache = sess.run(hidden2, feed_dict={X: mnist.train.images})
    h2_cache_test = sess.run(hidden2, feed_dict={X: mnist.test.images}) # not shown in the book

    for epoch in range(n_epochs):                                               # This training operation does not touch the lower frozen layers.
        shuffled_idx = np.random.permutation(mnist.train.num_examples)          #
        hidden2_batches = np.array_split(h2_cache[shuffled_idx], n_batches)
        y_batches = np.array_split(mnist.train.labels[shuffled_idx], n_batches)
        for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
            sess.run(training_op, feed_dict={hidden2:hidden2_batch, y:y_batch})

        accuracy_val = accuracy.eval(feed_dict={hidden2: h2_cache_test, y: mnist.test.labels})
        print("Epoch:" epoch, "Test accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_new_model_final.ckpt")

# Using faster optimizers.
# Momentum optimizer:
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
# Nesterov accelerated gradient:
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9
                use_nesterov=True)                                              # Just one simple tweak!!
# AdaGrad:
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
# RMS Prop:
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9,
                decay=0.9, epsilon=1e-10)
# Adam Optimizer (Adaptive moment estimation; combines Momentum & RMSProp):
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Learning rate scheduling (example using exponential_decay(), which implements
# exponential scheduling):
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 0.1
global_step = tf.Variable(0, trainable=False, name="global_step")
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(loss, global_step=global_step)

# ------------ Part 3: Regularization of large neural networks -----------------
# L1 and L2 Regularization:
# After constructing the neural network...
w_1 = tf.get_default_graph().get_tensor_by_name("hidden_1/kernel:0")            # Hidden layer 1 weights
w_2 = tf.get_default_graph().get_tensor_by_name("outputs/kernel:0")             # Output layer weights
scale = 0.001

with tf.name_scope("loss"):
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    base_loss = tf.reduce_mean(x_entropy, name="avg_x_entropy")
    reg_losses = tf.reduce_sum(tf.abs(w_1)) + tf.reduce_sum(tf.abs(w_2))
    loss = tf.add(base_loss, scale*reg_losses, name="loss")
# But the above is not good for larger networks with a lot of hidden layers.
# There is a better way...
my_dense_layer = partial(
            tf.layer.dense, activation=tf.nn.relu,
            kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))

with tf.name_scope("dnn"):
    hidden_1 = my_dense_layer(X, n_hidden_1, name="hidden_1")
    hidden_2 = my_dense_layer(hidden_1, n_hidden_2, name="hidden_2")
    logits = my_dense_layer(hidden_2, n_outputs, activation=None, name="outputs")

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)              # Add the regularization losses to the overall loss
loss = tf.add_n([base_loss] + reg_losses, name="loss")

# Dropout:
# ...create neural network...
training = tf.placeholder_with_default(False, shape=(), name="training")

dropout_rate = 0.5
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope("dnn"):
    hidden_1 = tf.layers.dense(X_drop, n_hidden_1, activation=tf.nn.relu, name="hidden_1")
    hidden_1_drop = tf.layers.dropout(hidden_1, dropout_rate, training=training)

    hidden_2 = tf.layers.dense(hidden_1_drop, n_hidden_2, activation=tf.nn.relu, name="hidden_2")
    hidden_2_drop = tf.layers.dropout(hidden_2, dropout_rate, training=training)

    logits = tf.layers.dense(hidden_2_drop, n_outputs, name="outputs")

# Max-Norm Regularization:
threshold = 1.0
weights = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")         # Get a handle on the weights in the first hidden layers
clipped_weights = tf.clip_by_norm(weights, clip_norm=threshold, axes=1)         # Creates an operation that will clip weights along second axis so each row vector has a max norm of 1.0
clip_weights = tf.assign(weights, clipped_weights)                              # Assigns the clipped weights to the weights variable

sess.run(training_op, feed_dict={X: X_batch, y: y_batch})                       # Apply the above operation after each training step
clip_weights.eval()


def max_norm_regularizer(threshold, axes=1, name="max_norm",                    # A cleaner solution to the above is to create a function
                         collection="max_norm"):
    """Creates a parameterized max_norm() function that can be used like any
    other regularizer."""
    def max_norm(weights):
        """..."""
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None # there is no regularization loss term
    return max_norm

max_norm_reg = max_norm_regularizer(threshold=1.0)                              #

with tf.name_scope("dnn"):                                                      #
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                              kernel_regularizer=max_norm_reg, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu,
                              kernel_regularizer=max_norm_reg, name="hidden2")
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

clip_all_weights = tf.get_collection("max_norm")                                # Fetch clipping operations and run after each training step

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            sess.run(clip_all_weights)
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        print("Epoch:" epoch, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# Data Augmentation:
# See book! (page 311) - NB TensorFlow has several image manipulation operations
