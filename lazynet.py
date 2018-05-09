

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
import tools


def dropout_with_drop_value(prev_matrix, alpha):
    elements = tf.size(prev_matrix)
    dist = tf.Binomial(total_count=elements, probs=alpha)
    mask = tf.fill(tf.shape(prev_matrix), 1.0)
    one_mask = tf.fill(tf.shape(prev_matrix), 1.0)
    #TODO: make values according prob
    dist.prob(mask)
    result = mask + tf.multiply((one_mask - mask), prev_matrix)
    return result

def get_actual_matrix(prev_matrix, inter_matrix, alpha):
    a = dropout_with_drop_value(prev_matrix, alpha)
    #TODO: should be the same shape
    act_matrix = tf.minimum(a, inter_matrix)
    return act_matrix

def change_shape(matrix, size):
    """
    tf.concat(3*3*3, 3*3*3, axis=3)=> [3*3*6]

    tf.slice (in_tensor, begin=[],size=[])
    tf.slice(in,begin[2-axis,1-axis,3 -axis] size[1,1,1](allocate memory))

    t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])
    # 0-2 dim 1-1 dim 2-3 dim
    x = tf.slice(t, [2, 0, 0], [1, 1, 3])=> [5,5,5]
    x = tf.slice(t, [2, 1, 0], [1, 1, 3])=> [6,6,6]
    x = tf.slice(t, [1, 1, 0], [1, 1, 3])=> [4,4,4]
    :return:
    """
    #concat
    result = matrix
    for i in range(0, size):
        tf.concat([result, matrix],3)
    return result

def create_tensor(t,s):
    arr = [[t for x in range(s)] for y in range(s)]
    out = tf.convert_to_tensor(arr)
    return out

x = tf.placeholder(tf.float32, (None, 28*28))
y = tf.placeholder(tf.float32, (None, 10))
keep_prob = tf.placeholder(tf.float32)

activations = []
activation_maps = []

x_image = tf.reshape(x, (-1, 28, 28, 1))
a_map_init = tf.reshape(tf.ones([28, 28, 1]), [-1, 28, 28, 1])

activations.append(x_image)
activation_maps.append(a_map_init)

classCount = 10
NORM = 10
N = 8
filters = [N, int(1.5*N), 2*N, 3*N, 4*N]
squeezes = [int(N/4), int(0.375*N), int(0.5*N), int(0.75*N), N]

print ("init:", activations[-1])

def fireblock(inputs,  expandTo, squeezeTo=4):
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


test = True
t = 0.2

with tf.name_scope('fire1' + str(N)):
    activations.append(tf.multiply(activations[-1], activation_maps[-1]))# AM train-time
    fireblock(activations[-1], filters[0], squeezes[0])
    h = tf.reshape(tf.reduce_mean(tf.nn.sigmoid(activations[-1]), 3), [-1, 28, 28, 1]) #IM 28x28x1
    if test:
        IM = tf.to_float(h > t)
        activation_maps.append(tf.minimum(IM, activation_maps[-1]))# min(IM,PM)
    else:
        activation_maps.append(tf.multiply(h, activation_maps[-1]))# IM* PM

with tf.name_scope('maxpool1'):
    h = tf.nn.max_pool(activations[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)
    h = tf.nn.max_pool(activation_maps[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activation_maps.append(h)

with tf.name_scope('fire2' + str(1.5*N)):
    activation_maps.append(change_shape(activation_maps[-1], 2*N))
    activations.append(tf.multiply(activations[-1], activation_maps[-1]))
    fireblock(activations[-1], filters[1], squeezes[1])
    h = tf.reshape(tf.reduce_mean(tf.nn.sigmoid(activations[-1]),3), [-1, 14, 14, 1])
    if test:
        IM = tf.to_float(h > t)
        activation_maps.append(tf.minimum(IM, activation_maps[-1]))
    else:
        activation_maps.append(tf.multiply(h, activation_maps[-1]))

with tf.name_scope('maxpool2'):
    h = tf.nn.max_pool(activations[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)
    h = tf.nn.max_pool(activation_maps[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activation_maps.append(h)

with tf.name_scope('fire3' + str(2 * N)):
    activation_maps.append(change_shape(activation_maps[-1],3*N))
    activations.append(tf.multiply(activations[-1], activation_maps[-1]))
    fireblock(activations[-1], filters[2], squeezes[2])
    h = tf.reshape(tf.reduce_mean(tf.nn.sigmoid(activations[-1]), 3), [-1, 7, 7, 1])
    if test:
        IM = tf.to_float(h > t)
        activation_maps.append(tf.minimum(IM, activation_maps[-1]))
    else:
        activation_maps.append(tf.multiply(h, activation_maps[-1]))

with tf.name_scope('maxpool3'):
    h = tf.nn.max_pool(activations[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)
    h = tf.nn.max_pool(activation_maps[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activation_maps.append(h)

with tf.name_scope('fire4' + str(3* N)):
    activation_maps.append(change_shape(activation_maps[-1],4*N))
    activations.append(tf.multiply(activations[-1], activation_maps[-1]))
    fireblock(activations[-1], filters[3], squeezes[3])
    h = tf.reshape(tf.reduce_mean(tf.nn.sigmoid(activations[-1]), 3), [-1, 4, 4, 1])
    if test:
        IM = tf.to_float(h > t)
        activation_maps.append(tf.minimum(IM, activation_maps[-1]))
    else:
        activation_maps.append(tf.multiply(h, activation_maps[-1]))

with tf.name_scope('maxpool4'):
    h = tf.nn.max_pool(activations[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)
    h = tf.nn.max_pool(activation_maps[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activation_maps.append(h)

with tf.name_scope('fire5' + str(4* N)):
    activation_maps.append(change_shape(activation_maps[-1],6*N))
    activations.append(tf.multiply(activations[-1], activation_maps[-1]))
    fireblock(activations[-1], filters[4], squeezes[4])
    h = tf.reshape(tf.reduce_mean(tf.nn.sigmoid(activations[-1]), 3), [-1, 2, 2, 1])
    if test:
        IM = tf.to_float(h > t)
        activation_maps.append(tf.minimum(IM, activation_maps[-1]))
    else:
        activation_maps.append(tf.multiply(h, activation_maps[-1]))

with tf.name_scope('maxpool5'):
    print(activations[-1])
    h = tf.nn.max_pool(activations[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)
    h = tf.nn.max_pool(activation_maps[-1], [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    activation_maps.append(h)

with tf.name_scope('dense'):
    # second parameter 64-batch size
    a_flat = tf.reshape(activations[-1], [100, 2*4*N])
    dense = tf.layers.dense(inputs=a_flat, units=10, activation=tf.nn.sigmoid)
    activations.append(dense)

with tf.name_scope('logist'):
    y_conv = tf.nn.softmax(activations[-1])
    print("y_conv: ", tf.argmax(y_conv, 1))
    print("y_answers", y)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=activations[-1], labels=y)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
