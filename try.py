import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

def run_cnn():
 mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 learning_rate = 0.0001
 epochs = 10
 batch_size = 50


x = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 784])
x_shaped = tf.compat.v1.reshape(x, [-1, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 10])

layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2],
name='layer2')

flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])
wd1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1000], stddev=0.03), name='wd1')
d1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)