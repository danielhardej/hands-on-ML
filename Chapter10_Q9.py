from __future__ import division, print_function, unicode_literals               # Needed for interations/loops to work properly!!!
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

# ----------------------- Training a deep MLP on MNIST -------------------------
# Goals: achieve over 98% accuracy; add save checkpoint; restore; add summaries;
# plot learning curves on TensorBoard

# Import the data from the TensorFlow tutorial sets - all is scaled and shuffled
mnist = input_data.read_data_sets("/tmp/data/")

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

n_inputs = 28*28  # MNIST
n_hidden_1 = 300
n_hidden_2 = 100
n_outputs = 10

# ------- Step 1: construction phase; build TensorFlow graph -------
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")                # Use placeholders to represent the training data with dimensions size_training_data x n_features
y = tf.placeholder(tf.int64, shape=(None), name="y")                            # ...and the targets. Placeholder X also acts as the input layer - replaced with one training batch at a time.

with tf.name_scope("dnn"):
    hidden_1 = tf.layers.dense(X, n_hidden_1, name="hidden_1", activation=tf.nn.relu)
    hidden_2 = tf.layers.dense(hidden_1, n_hidden_2, name="hidden_2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden_2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)     # This funciton is equivalent to applying softmax activaiton, then cross entropy
    loss = tf.reduce_mean(x_entropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

learning_rate = 0.01
with tf.name_scope("training"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):                                                     # Define method of evaluating performance of model
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()                                        # Define the node to initialize all variables
saver = tf.train.Saver()

# Define directory to write TensorBoard logs to:
def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

logdir = log_dir("mnist_dnn")

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())             # Creates the file writer to create the TensorBoard logs

X_valid = mnist.validation.images
y_valid = mnist.validation.labels

m, n = X_train.shape
# ---------------------- Step 2: Execution phase ----------------------

n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_deep_mnist_model"

best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50

with tf.Session() as sess:
        if os.path.isfile(checkpoint_epoch_path):                               # if the checkpoint file exists, restore the model and load the epoch number
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run([accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)
        if epoch % 5 == 0:
            print("Epoch:", epoch, "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100), "\tLoss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break

# os.remove(checkpoint_epoch_path)

with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})

print("Accuracy: ", accuracy)
