from __future__ import division, print_function, unicode_literals               # Needed for interations/loops to work properly!!!
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data
# Import the data from the TensorFlow tutorial sets - all is scaled and shuffled
mnist = input_data.read_data_sets("/tmp/data/")

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

# ------------------------- Training an MLP with TF ----------------------------
# NB MLP created using TensorFlow high-level API, tensorflow.contrib

# config = tf.contrib.learn.RunConfig(tf_random_seed=42)
#
# feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
# # Create a DNN with two layers: one with 300 units, one with 100; softmax output
# # layer with 10 output neurons; output layer uses Softmax function;
# # DNNClassifier creates all neurons as ReLUs apart from output layer.
# dnn_classifier = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=10, feature_columns=feature_cols)
# dnn_classifier = tf.contrib.learn.SKCompat(dnn_classifier)                      #
# dnn_classifier.fit(X_train, y_train, batch_size=50, steps=40000)                #
#                                                                                 #
# y_pred = dnn_classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred["classes"])
# print("MLP accuracy:", accuracy)

# ------------------------- Training a DNN with TF -----------------------------
# Mini-batch gradient descent on the MNIST dataset
# Step 1: construction phase; build TensorFlow graph

n_inputs = 28*28  # MNIST
n_hidden_1 = 300
n_hidden_2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")                # Use placeholders to represent the training data with dimensions size_training_data x n_features
y = tf.placeholder(tf.int64, shape=(None), name="y")                            # ...and the targets. Placeholder X also acts as the input layer - replaced with one training batch at a time.
# Now create the two hidden layers and the output layer:
def neuron_layer(X, n_neurons, name, activation=None):
    """Creates a layer of the neur network, based on specified inputs, number of
    neurons, the activation funcion, and the name of the layer."""
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)                                          # Truncated (Gaussian) normal ensures there are no large weights, which slow down training
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        w = tf.Variable(init, name="kernel")                                    # w stores the weights as a matrix; named "kernel"
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        z = tf.matmul(X, w) + b
        if activation is not None:
            return activation(z)
        else:
            return z

with tf.name_scope("dnn"):                                                      # Now create the DNN using the neuron_layer function defined above.
    hidden_1 = neuron_layer(X, n_hidden_1, name="hidden_1", activation=tf.nn.relu)
    hidden_2 = neuron_layer(hidden_1, n_hidden_2, name="hidden_2", activation=tf.nn.relu)
    logits = neuron_layer(hidden_2, n_outputs, name="outputs")                  # The logits layer is the output before going into the softmax function

# An alternative: using the dense() function, rather then defining our own neural network layer
# with tf.name_scope("dnn"):
#     hidden_1 = tf.layers.dense(X, n_hidden_1, name="hidden_1", activation=tf.nn.relu)
#     hidden_2 = tf.layers.dense(hidden_1, n_hidden_2, name="hidden_2", activation=tf.nn.relu)
#     logits = tf.layers.dense(hidden_2, n_outputs, name="outputs")

with tf.name_scope("loss"):                                                     # Define the cost functionto train the network using cross-entropy
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)     # This funciton is equivalent to applying softmax activaiton, then cross entropy
    loss = tf.reduce_mean(x_entropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):                                                    # Define the optimizer to train the model
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)                # use gradient descent to tweak model params and optimize cost funciton
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):                                                     # Define method of evaluating performance of model
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()                                        # Define the node to initialize all variables
saver = tf.train.Saver()                                                        # Saver to save the trained model to disk

# Step 2: execution phase; run the graph; train model
n_epochs = 40
batch_size = 50

with tf.Session() as sess:
    init.run()
    epoch_list = []
    train_acc_list = []
    test_acc_list = []
    for epoch in range(n_epochs):
        for i in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})

        epoch_list.append(epoch)
        train_acc_list.append(acc_train)
        test_acc_list.append(acc_test)

        print("Epoch:", epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")

plt.plot(epoch_list, train_acc_list, "b")
plt.plot(epoch_list, test_acc_list, "r")
plt.legend((train_acc_list, test_acc_list), ("Training accuracy", "Test accuracy"))
plt.show()

# Restoring and using the trained nerual network
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt")
    X_new_scaled = mnist.test.images[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)


print("Predicted classes:", y_pred)
print("Actual classes:   ", mnist.test.labels[:20])
