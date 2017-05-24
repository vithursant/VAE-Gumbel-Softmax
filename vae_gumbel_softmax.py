from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import seaborn as sns
import os
from tqdm import trange

from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

sns.set_style('whitegrid')

# Define the different distributions
distributions = tf.contrib.distributions

bernoulli = distributions.Bernoulli
onehot_categorical = distributions.OneHotCategorical
relaxed_onehot_categorical = distributions.RelaxedOneHotCategorical

# Define Directory Parameters
flags = tf.app.flags
flags.DEFINE_string('data_dir', os.getcwd() + '/data/', 'Directory for data')
flags.DEFINE_string('log_dir', os.getcwd() + '/log/', 'Directory for logs')

# Define Model Parameters
flags.DEFINE_integer('batch_size', 100, 'Minibatch size')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate') 
flags.DEFINE_integer('num_classes', 10, 'Number of classes')
flags.DEFINE_integer('num_cat_dists', 200, 'Number of categorical distributions') # num_cat_dists//num_calsses
flags.DEFINE_float('init_temp', 1.0, 'Initial Temperature')
flags.DEFINE_bool('straight_through', False, 'Straight-through Gumbel-Softmax')
flags.DEFINE_string('kl_type', 'relaxed', 'Kullback-Leibler Divergence (relaxed or categorical)')
flags.DEFINE_bool('learn_temp', False, 'Learn temperature parameter')

FLAGS = flags.FLAGS


def train():
    x = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 784), name = 'x')
    
    # We use the MNIST dataset with fixed binarization for training and evaluation, which is common
    net = tf.cast(tf.random_uniform(tf.shape(x)) < x, x.dtype) 
    net = slim.stack(net, slim.fully_connected, [512, 256])

    logits_y = tf.reshape(slim.fully_connected(net, FLAGS.num_classes*FLAGS.num_cat_dists, activation_fn=None), [-1, FLAGS.num_cat_dists, FLAGS.num_classes])
    tau = tf.Variable(FLAGS.init_temp, name = "temperature", trainable = FLAGS.learn_temp)
    q_y = relaxed_onehot_categorical(tau, logits_y)
    y = q_y.sample()

    if FLAGS.straight_through:
        y_hard = tf.cast(tf.one_hot(tf.argmax(y, -1), FLAGS.num_classes), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
   
    net = slim.flatten(y)
    net = slim.stack(net, slim.fully_connected, [256, 512])
    
    logits_x = slim.fully_connected(net, 784, activation_fn=None)
    p_x = bernoulli(logits = logits_x)
    x_mean = p_x.mean()

    recons = tf.reduce_sum(p_x.log_prob(x), 1)
    logits_py = tf.ones_like(logits_y) * 1./FLAGS.num_classes

    if FLAGS.kl_type == 'categorical' or FLAGS.straight_through:
        p_cat_y = onehot_categorical(logits = logits_py)
	q_cat_y = onehot_categorical(logits = logits_y)
	KL_qp = distributions.kl(q_cat_y, p_cat_y)
    else:
	p_y = relaxed_onehot_categorical(tau, logits = logits_py)
	KL_qp = q_y.log_prob(y) - p_y.log_prob(y)

    KL = tf.reduce_sum(KL_qp, 1)
    mean_recons = tf.reduce_mean(recons)
    mean_KL = tf.reduce_mean(KL)
    loss = -tf.reduce_mean(recons - KL)

    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

    data = []
    mnist = input_data.read_data_sets(FLAGS.data_dir + '/MNIST', one_hot=True)

    with tf.train.MonitoredSession() as sess:
	train_epoch = trange(50000, desc='Loss', leave=True)
	for i in train_epoch:
	    batch = mnist.train.next_batch(FLAGS.batch_size)
	    res = sess.run([train_op, loss, tau, mean_recons, mean_KL], {x : batch[0]})
      	    if i % 100 == 1:
		data.append([i] + res[1:])
	    if i % 1000 == 1:
		print('Step %d, Loss: %0.3f' % (i, res[1]))
	# end training - do an eval
	batch = mnist.test.next_batch(FLAGS.batch_size)
	np_x = sess.run(x_mean, {x : batch[0]})

    data = np.array(data).T

    f, axarr = plt.subplots(1, 4, figsize=(18, 6))
    axarr[0].plot(data[0], data[1])
    axarr[0].set_title('Loss')

    axarr[1].plot(data[0], data[2])
    axarr[1].set_title('Temperature')

    axarr[2].plot(data[0], data[3])
    axarr[2].set_title('Recons')

    axarr[3].plot(data[0], data[4])
    axarr[3].set_title('KL')


    tmp = np.reshape(np_x, (-1, 280, 28))
    img = np.hstack([tmp[i] for i in range(10)])

    plt.imshow()
    plt.grid('off')

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.data_dir)
    train()

if __name__=="__main__":
    tf.app.run()
