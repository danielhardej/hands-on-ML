# ------------------------------------------------------------------------------
#                   Chapter 13:
# ------------------------------------------------------------------------------
from __future__ import division, print_function, unicode_literals
import numpy as np
import os
import tensorflow as tf
from sklearn.datasets import load_sample_images

# -------------------------- Convolutional Layers ------------------------------
china = load_sample_image("china.jpg")                                          # loads two sample images
flower = load_sample_image("flower.jpg")

dataset = np.array([china, flower], dtype=np.float32)                           # Generate the dataset from the images
batch_size, height, width, channels = dataset.shape                             # Assign the tensor shape using the dataset

filters = np.zeros(shape=(7,7,channels,2), dtype=np.float32)                    # Create a 7 x 7 filter
filters[:, 3, :, 0] = 1                                                         # Vertical line filter
filters[3, :, :, 0] = 1                                                         # Horizontal line filter

X = tf.placeholder(tf.float32, shape=(None, height, width, channnels))
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")       # Create the 2D convolutional layer; apply filter to images; zero padding; stride 2

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

# ----------------------------- Pooling Layers ---------------------------------
# Implemeting a max-pooling layer:
china = load_sample_image("china.jpg")                                          # loads two sample images
flower = load_sample_image("flower.jpg")

dataset = np.array([china, flower], dtype=np.float32)                           # Generate the dataset from the images
batch_size, height, width, channels = dataset.shape                             # Assign the tensor shape using the dataset

X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")   # ksize contains the kernal shape along all four dimensions of the input tensor [batch_size, height, width, channels]
                                                                                # NB and average-pooling layer can be creted using avg_pool()
with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})
