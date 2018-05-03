import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

x = tf.placeholder(tf.float32, (None, 28*28))
y = tf.placeholder(tf.float32, (None, 10))
keep_prob = tf.placeholder(tf.float32)

activations = []
activation_maps = []

x_image = tf.reshape(x, (-1, 28, 28, 1))
a_map_init = tf.ones([28, 28, 1])

activations.append(x_image)
activation_maps.append(a_map_init)

classCount = 10
NORM = 100
N = 6
filters = [N, int(1.5*N), 2 * N, 3 * N, 4 * N]

def fireblock(inputs,  expandTo, squeezeTo=4,):
    h = squeeze(inputs, squeezeTo)
    h = expand(h, expandTo)
    h = tf.clip_by_norm(h, NORM)
    activations.append(h)

def squeeze(inputs, squeezeTo):
    with tf.name_scope('squeeze'):
        inputSize = inputs.get_shape().as_list()[3]
        w = tf.Variable(tf.truncated_normal([1, 1, inputSize, squeezeTo]))
        h = tf.nn.leaky_relu(tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME'), alpha=0.05)
    return h

def expand(inputs, expandTo):
    with tf.name_scope('expand'):
        squeezeTo = inputs.get_shape().as_list()[3]
        w = tf.Variable(tf.truncated_normal([1, 1, squeezeTo, expandTo]))
        h1x1 = tf.nn.leaky_relu(tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME'), alpha=0.05)
        w = tf.Variable(tf.truncated_normal([3, 3, squeezeTo, expandTo]))
        h3x3 = tf.nn.leaky_relu(tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME'), alpha=0.05)
        h = tf.concat([h1x1, h3x3], 3)
    return h


with tf.name_scope('fire1' + str(N)):
    print(activations[-1])
    fireblock(activations[-1], filters[0])
    print(activations[-1])

with tf.name_scope('maxpool1'):
    print(activations[-1])
    h = tf.nn.max_pool(activations[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)
    print(activations[-1])

with tf.name_scope('fire2' + str(1.5*N)):
    print(activations[-1])
    fireblock(activations[-1], filters[1])
    print(activations[-1])

with tf.name_scope('maxpool2'):
    print(activations[-1])
    h = tf.nn.max_pool(activations[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)
    print(activations[-1])

with tf.name_scope('fire3' + str(2 * N)):
    fireblock(activations[-1], filters[2])

with tf.name_scope('maxpool3'):
    print(activations[-1])
    h = tf.nn.max_pool(activations[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)
    print(activations[-1])

with tf.name_scope('fire4' + str(3* N)):
    print(activations[-1])
    fireblock(activations[-1], filters[3])
    print(activations[-1])

with tf.name_scope('maxpool4'):
    print(activations[-1])
    h = tf.nn.max_pool(activations[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)
    print(activations[-1])

with tf.name_scope('fire5' + str(4* N)):
    print(activations[-1])
    fireblock(activations[-1], filters[4])
    print(activations[-1])

with tf.name_scope('maxpool5'):
    print(activations[-1])
    h = tf.nn.max_pool(activations[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)
    print(activations[-1])

with tf.name_scope('dense'):
    print(activations[-1])
    a_flat = tf.reshape(activations[-1], [64, 8*N])
    print(a_flat)
    dense = tf.layers.dense(inputs=a_flat, units=10, activation=tf.nn.relu)
    print(dense)
    activations.append(dense)
    print("final_OUT:", activations[-1])

with tf.name_scope('logist'):
    y_conv = tf.nn.softmax(activations[-1])
    print("y_conv: ", tf.argmax(y_conv, 1))
    print("y_answers",y)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=activations[-1], labels=y)
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

