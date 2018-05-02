from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import tensorflow as tf


N=10
tf.logging.set_verbosity(tf.logging.INFO)


"""def find_blocks():
    start=True
    left_top=tuple(-1,-1)
    right_bottom=tuple(-1,-1)
    left=0
    right=0
    top=0
    bottom=0
    for i in range(am.shape[0]):
        bottom=i
        for j in range(am.shape[1]):
            if (am[i,j]>0):
                if start:
                    left_top=tuple(i,j)
                    right=j
        if ()"""
'''def cut_regions(tensor, AM):
    array = tensor.eval()
    scale = array.shape
    am = AM.eval()

    return find_blocks(array, am)'''

#init activation matrix
activ_map = np.zeros(shape=[784])
activ_map = np.ones_like(activ_map)
AM = tf.convert_to_tensor(activ_map, tf.float32)

function_to_tensor = lambda x: 1/(1+math(x))
#out= tf.map_fn(function_to_tensor, input)

def fire_block(input, s_depth, e_depth):

    #squeeze layer
    net = tf.layers.conv2d(
        inputs=input,
        filters=s_depth,
        kernel_size=[1,1],
        padding="same",
        activation=tf.nn.relu,
    )
    print("net s11", net.get_shape())
    #expand layer
    net1 = tf.layers.conv2d(
        inputs=net,
        filters=e_depth,
        kernel_size=[1,1],
        padding="same",
        activation=tf.nn.relu,
    )
    print("net e11", net1.get_shape())
    net2 = tf.layers.conv2d(
        inputs=net,
        filters=e_depth,
        kernel_size=[3,3],
        padding="same",
        activation=tf.nn.relu,
    )
    out = tf.multiply(net1, net2)
    print("net e33", net2.get_shape())
    print("out ", out.get_shape())
    return out

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  activ_map = np.zeros(shape=[784])
  activ_map = np.ones_like(activ_map)
  #AM = tf.convert_to_tensor(activ_map, tf.float32)
  #AF = tf.multiply(AM, tf.map_fn(function_to_tensor, pool1))

  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = fire_block(input_layer, 2 * N, 2 * N)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  tf.sigmoid(pool1)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = fire_block(pool1, 1.5 * N, 1.5 * N)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  tf.sigmoid(pool2)

  # Convolutional Layer #3 and Pooling Layer #3
  conv3 = fire_block(pool2, 3 * N, 3 * N)
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  tf.sigmoid(pool3)

  # Convolutional Layer #4 and Pooling Layer #4
  conv4 = fire_block(pool3, 4 * N, 4 * N)
  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
  tf.sigmoid(pool4)
  print("pool4", pool4.get_shape())

  # Dense Layer
  pool2_flat = tf.reshape(pool3, [-1, 1 * 1 * N/2 * 4])
  print("pool2_flat", pool2_flat.get_shape())
  dense = tf.layers.dense(inputs=pool2_flat, units=N, activation=tf.nn.relu)
  print("dense", dense.get_shape())
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  # Logits Layer
  logits = tf.layers.dense(inputs=dense, units=10)

  print("logits", logits.get_shape())
  print("labels", labels.get_shape())

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
  mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

# {'loss': 0.10483072, 'global_step': 20000, 'accuracy': 0.9681} # tutorial implementation
#{'loss': 2.3010862, 'accuracy': 0.1135, 'global_step': 20000}
if __name__ == "__main__":
  tf.app.run()

# #1 {'loss': 2.301103, 'global_step': 20001, 'accuracy': 0.1135}
# #2{'loss': 2.301065, 'accuracy': 0.1135, 'global_step': 52425}