from __future__ import division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Constructing a graph and executing it...
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
    print(result)

# Managing multple independant graphs and creating new graphs
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

tf.reset_default_graph()

# Node value lifecycles

w = tf.constant(3)
x = w + 2
y = x + 5
z = y * 3
with tf.Session() as sess:
    print(y.eval())
    print(z.eval())
# But the above is bad code - it does not reuse the values of x and w when it
# computes y and z. It evaluates the code to compute w and x twice.
# But below, the code is much more efficient, computing each variable only once!
# It does this by evaluating y and z in the same line.
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)

# Linear Regression with TensorFlow - tested on the California housing data
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_with_bias = np.c_[np.ones((m,1)), housing.data]

X = tf.constant(housing_data_with_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_val = theta.eval()
# the above uses the Normal Equation to evaluate the theta values. The code
# below implements batch Gradient Descent instead.
n_epochs = 1000
learning_rate = 0.01

scaler = StandardScaler()
# Better feature scaling --- min MSE = 0.524321
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_with_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
# Worse feature scaling --- min MSE = 4.80326
#scaled_housing_data_with_bias = scaler.fit_transform(housing_data_with_bias.astype(np.float32))

X = tf.constant(scaled_housing_data_with_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")

theta  = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = (2/m)*tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate*gradients)                 # Set the training optimizer: theta(next) = theta - learning_rate*gradMSE

init = tf.global_variables_initializer()                                        # Initialize all of the variables

with tf.Session() as sess:                                                      # Set the default session to run
    sess.run(init)

    for epoch in range(n_epochs):
        if (epoch % 100) == 0:
            print("Epoch: ", epoch, "MSE=", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    print(best_theta)

# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
# training_op = optimizer.minimize(mse)

# Now lets test out autodiff...
# Using gradient descent requires that we caluculate the gradients of the cost
# function - in this case, the MSE. Easy in LinReg, but if we were using big
# neural nets then it would be a bit shit.
# Here comes TensorFlow autodiff to the rescue...
# gradients = tf.gradients(mse, [theta])
# Now lets look at optimizers...
# If we replace training_op = ... and gradients = ... in the previous code with:
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
# training_op = optimizer.minimize(mse)

# Now lets look at how to feed data to the training algorithm. We can modify the
# previous code to implement mini-batch gradient descent...
# to do this, we need to find a way to replace X and y with a new batch at every
# iteration with the next mini-batch. We to this with placeholder nodes...
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
print(B_val_1)
print(B_val_2)

# # For the code above, we tweak only the definition of X and y...
# X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
# y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
#
# # And then define the batch size and the number of batches too...
# batch_size = 100
# n_batches = int(np.ceil(m/batch_size))
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# # In execution phase, feed in the mini-batches one by one...
# def fetch_batch(epoch, batch_index, batch_size):
#     np.random.seed(epoch * n_batches + batch_index)
#     indices = np.random.randint(m, size=batch_size)
#     X_batch = scaled_housing_data_with_bias[indices]
#     y_batch = housing.target.reshape(-1, 1)[indices]
#     return X_batch, y_batch
# # Then run the session, and provide the X and y values via the feed_dict parameter
# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(n_epochs):
#         for batch_index in range(n_batches):
#             X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
#             save_path = saver.save(sess, "/tmp/my_model.ckpt")
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#
#     best_theta = theta.eval()
#     print("Best theta val for Mini-batch GD: ", best_theta)
#     save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
#     print("Model successfully saved!")

# Now something else important: Saving models!
# Create the Saver node at the end of the construction phase (after all
# variables have been created)...
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# Then, during the execution phase, call the save() method whenever you want to
# save the model.
# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(n_epochs):
#         if epoch % 100 == 0:
#             print("Epoch", epoch, "MSE =", mse.eval())
#             save_path = saver.save(sess, "/tmp/my_model.ckpt")
#         sess.run(training_op)
#
#     best_theta = theta.eval()
#     save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
# Restoring the model is easy too. Just create a Saver at the end of the
# execution phase, but instead of initializing variables with the init node,
# we call the restore() method at the start of execution.
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, "tmp/my_model_final.ckpt")
#     best_theta_restored = theta.eval()
#     ...
# Say you need more control, and want to be able to save under different variable
# names. We can specify what variables to save and restore, and what names to use.
# saver = tf.train.Saver({"Weights": theta})
# # You can also load the graph structure using:
# saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")
# theta = tf.get_default_graph().get_tensor_by_name("theta:0")
# Then just do...
# with tf.Session() as sess:
#     saver.restore(sess, "tmp/my_model_final.ckpt")
#     ...
# This enables you to restore the graph structure AND variables vals!!!

# Okay, now we check out TensorBoard!
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_with_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]
    return X_batch, y_batch


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.close()

# Okay now we introduce name scopes
# These are used in neural networks, where the graph can become cluttered with
# thousands of nodes! We avoid this with name scopes.
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

# -------------------------------- Modularity ----------------------------------
# Lets look at code that adds the outputs of two ReLUs...
n_features = 3
# X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
#
# w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
# w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
# b1 = tf.Variable(0.0, name="bias1")
# b2 = tf.Variable(0.0, name="bias2")
#
# z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
# z2 = tf.add(tf.matmul(X, w2), b2, name="z2")
#
# relu1 = tf.maximum(z1, 0., name="relu1")
# relu2 = tf.maximum(z2, 0., name="relu2")
#
# output = tf.add(relu1, relu2, name="output")
# That's okay for small networks with only two ReLUs, but that is never the case
# in practice - in real life, networks are much bigger and using this method is
# completely impractical...
# Fortunately, in TensorFlow we can create a function that builds a network!
# def relu(X):
#     w_shape = (int(X.get_shape()[1]), 1)
#     w = tf.Variable(tf.random_normal(w_shape), name="weights")
#     b = tf.Variable(0.0, name="bias")
#     z = tf.add(tf.matmul(X, w), b, name="z")
#     return tf.maximum(z, 0., name="relu")
#
# n_features = 3
# X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
# relus = [relu(X) for i in range(5)]
# output = tf.add(relus, name="output")

# But to make it even better, you can use name scopes - the following code will
# create a name scope for each relu, appending its index onto its name
def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="maximum")                         # NB Here we can also get the ReLUs to share variables! Compare with what is returned by
                                                                                # the previous function.
threshold = tf.Variable(0.0, name="threshold")                                  # Here we create the variable 'threshold', then pass it to the function! The threshold is
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")              # the shared variable!
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

# So this is fine if we only have one variable to control, but passing a number
# of shared variables can be a headache.
# To get around this, it's good to create a
# python dictionary containing all of the variables in the model, then pass it to
# each function. Yet another option is to create a class for each module.
# Yet another (perhaps better) option is to set the shared variables as an attribute
# of the relu() function the first time that you call it...
def relu(X):
    with tf.name_scope("relu"):
        if not hasattr(relu, "threshold"):
            relu.threshold = tf.Variable(0.0, name="threshold")
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, relu.threshold, name="max")

# But there's yet another option, leading to cleaner and more modular code:
# 1) use the get_variable() function to create the shared variable (or reuse it if it already exists)
# 2) control its behaviour with the attribute of the current variable_scope()
# with tf.variable_scope("relu"):
#     threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
#
# with tf.variable_scope("relu", reuse=True):
#     threshold = tf.get_variable("threshold")
#
# with tf.variable_scope("relu") as scope:
#     scope.reuse_variables()
#     threshold = tf.get_variable("threshold")

# All the pieces together:
def relu(X):
    """This code defines the relu() function, creates the relu threshold variable,
        builds 5 relus. It reuses the relu threshold variable for each"""
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):                                                 # Create the variable
    threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")
