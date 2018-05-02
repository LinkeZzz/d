from squeeze import *
import time
import numpy as np
#import cv2
import sys
sys.path.append("")
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()

merged = tf.summary.merge_all()
summaryWriter = tf.summary.FileWriter('./Tensorboard',sess.graph)

sess.run(tf.global_variables_initializer())
mnist = input_data.read_data_sets('D:\TensorFlow\MNIST\data', one_hot=True)

start_time = time.time()
print('Started')
for i in range(2500):
    batch = mnist.train.next_batch(64)
    if i%100 == 0:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, train_accuracy = sess.run([merged, accuracy],
                                    feed_dict={x: batch[0], y: batch[1],keep_prob: 1},
                                    options=run_options,
                                    run_metadata=run_metadata)
        summaryWriter.add_run_metadata(run_metadata, 'step%03d' % i)
        summaryWriter.add_summary(summary, i)
        #[train_accuracy] = sess.run([accuracy],
        #                            feed_dict={x: batch[0], y: batch[1],keep_prob: 1})
        print("step %d, training accuracy %g %f"%(i, train_accuracy,time.time()-start_time))
        start_time = time.time()
    train_step.run(feed_dict={x: batch[0], y: batch[1],keep_prob:.5})

def dream(layer = -1,ITERATIONS = 50):
  img_noise = np.random.uniform(size=(28,28))
  #img_noise = np.ones((28,28)) * .5
  total_image = None

  for channel in range(activations[layer].get_shape().as_list()[-1]):
    try:
      t_obj = activations[layer][:,:,:,channel]
    except:
      t_obj = activations[layer][:,channel]
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score,x)[0]
    img = img_noise.copy()
    img = img.reshape(1,784)

    for i in range(ITERATIONS):
      g,score = sess.run([t_grad,t_score],{x:img})
      g /= g.std()+1e-8
      step = 1
      img += g*step
    print(channel,score)

    img = (img-img.mean())/max(img.std(), 1e-4)*.1 + 0.5
    if total_image is None:
      total_image = img.reshape((28,28))
    else:
      total_image = np.hstack((total_image,img.reshape((28,28))))
  #cv2.imwrite('Total_%s.png'%layer,total_image * 255)

def dreamAll(ITERATIONS = 50):
  for i in range(len(activations)):
    print('Layer %d'%i)
    dream(i,ITERATIONS)