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
N = 5

def fireblock(inputs, squeezeTo, expandTo):
    h = squeeze(inputs, squeezeTo)
    h = expand(h, expandTo)
    h = tf.clip_by_norm(h, NORM)
    #activation_maps.append(tf.multiply(tf.nn.sigmoid(h), activation_maps[-1]))
    activations.append(tf.multiply(activation_maps[-1], h))
    d = tf.Print(activation_maps[-1], [activation_maps[-1]], "TENSOR PRINT: ")
    # activations.append(h)

def squeeze(inputs, squeezeTo):
    with tf.name_scope('squeeze'):
        inputSize = inputs.get_shape().as_list()[3]
        w = tf.Variable(tf.truncated_normal([1, 1, inputSize, squeezeTo]))
        h = tf.nn.relu(tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME'))
    return h

def expand(inputs, expandTo):
    with tf.name_scope('expand'):
        squeezeTo = inputs.get_shape().as_list()[3]
        w = tf.Variable(tf.truncated_normal([1, 1, squeezeTo, expandTo]))
        h1x1 = tf.nn.relu(tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME'))
        w = tf.Variable(tf.truncated_normal([3, 3, squeezeTo, expandTo]))
        h3x3 = tf.nn.relu(tf.nn.conv2d(inputs, w, [1, 1, 1, 1], 'SAME'))
        h = tf.concat([h1x1, h3x3], 3)
    return h

squeezes = [2 * N, 2 * N, int(1.5 * N), int(1.5 * N), 3 * N, 3 * N, 4 * N, 4 * N]
filters = [8 * N, 8 * N, 6 * N, 6 * N, 12 * N, 12 * N, 16 * N, 16 * N]

with tf.name_scope('conv1'):
    w = tf.Variable(tf.truncated_normal([3, 3, 1, 64]))
    h = tf.nn.relu(tf.nn.conv2d(activations[-1], w, [1, 2, 2, 1], 'SAME'))
    activation_maps.append(tf.nn.sigmoid(h))
    activations.append(tf.multiply(h, activation_maps[-1]))

with tf.name_scope('maxpool1'):
    h = tf.nn.max_pool(activations[-1], [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)

for i in range(0, 2):
    with tf.name_scope('fire' + str(i + 2)):
        fireblock(activations[-1], squeezes[i], filters[i])

with tf.name_scope('maxpool2'):
    h = tf.nn.max_pool(activations[-1], [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)

for i in range(2, 4):
    with tf.name_scope('fire' + str(i + 2)):
        fireblock(activations[-1], squeezes[i], filters[i])

with tf.name_scope('maxpool3'):
    h = tf.nn.max_pool(activations[-1], [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    activations.append(h)

for i in range(4, 7):
    with tf.name_scope('fire' + str(i + 2)):
        fireblock(activations[-1], squeezes[i], filters[i])

with tf.name_scope('dropout'):
    h = tf.nn.dropout(activations[-1], keep_prob)
    activations.append(h)

with tf.name_scope('conv10'):
    input_shape = activations[-1].get_shape().as_list()[3]
    w = tf.Variable(tf.truncated_normal([1, 1, input_shape, classCount]))
    h = tf.nn.relu(tf.nn.conv2d(activations[-1], w, [1, 1, 1, 1], 'SAME'))
    # activations.append(h)
    activation_maps.append(activation_maps.append(tf.multiply(tf.nn.sigmoid(h), activation_maps[-1])))
    activations.append(tf.multiply(h, activation_maps[-1]))

with tf.name_scope('avgpool'):
    input_shape = activations[-1].get_shape().as_list()[2]
    h = tf.nn.avg_pool(activations[-1], [1, input_shape, input_shape, 1], [1, 1, 1, 1], 'VALID')
    h = tf.squeeze(h, [1, 2])
    activations.append(h)

y_conv = tf.nn.softmax(activations[-1])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=activations[-1], labels=y)
train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)


