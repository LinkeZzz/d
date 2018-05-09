from lazynet import *
import time
import numpy as np
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
for i in range(2400):
    batch = mnist.train.next_batch(100)
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
sum=0
for i in range(0,100):
    sum+=accuracy.eval(feed_dict={x: mnist.test.images[100*i:100*i+100], y: mnist.test.labels[100*i:100*i+100]})
print("test acuuracy", sum/100)
saver = tf.train.Saver()
save_path = saver.save(sess, "/root/PycharmProjects/d/models/model.ckpt")
print("Model saved in path: %s" % save_path)
