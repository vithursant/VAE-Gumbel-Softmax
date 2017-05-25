from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import seaborn as sns
import os
import time
from tqdm import trange, tqdm

from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

sns.set_style('whitegrid')

# Define the different distributions
distributions = tf.contrib.distributions

bernoulli = distributions.Bernoulli

# Define current_time
current_time = time.strftime('%Y-%m-%d-%H-%M-%S')

# Define Directory Parameters
flags = tf.app.flags
flags.DEFINE_string('data_dir', os.getcwd() + '/data/', 'Directory for data')
flags.DEFINE_string('log_dir', os.getcwd() + '/log/', 'Directory for logs')
flags.DEFINE_string('checkpoint_dir', os.getcwd() + '/checkpoint/' + current_time, 'Directory for checkpoints')

# Define Model Parameters
flags.DEFINE_integer('batch_size', 100, 'Minibatch size')
flags.DEFINE_integer('num_iters', 50000, 'Number of iterations')
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate') 
flags.DEFINE_integer('num_classes', 10, 'Number of classes')
flags.DEFINE_integer('num_cat_dists', 200, 'Number of categorical distributions') # num_cat_dists//num_calsses
flags.DEFINE_float('init_temp', 1.0, 'Initial temperature')
flags.DEFINE_float('min_temp', 0.5, 'Minimum temperature')
flags.DEFINE_float('anneal_rate', 0.00003, 'Anneal rate')
flags.DEFINE_bool('straight_through', False, 'Straight-through Gumbel-Softmax')
flags.DEFINE_string('kl_type', 'relaxed', 'Kullback-Leibler divergence (relaxed or categorical)')
flags.DEFINE_bool('learn_temp', False, 'Learn temperature parameter')

FLAGS = flags.FLAGS

def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), 
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    
    return y
 
def encoder(x): 
    # Variational posterior q(y|x), i.e. the encoder (shape=(batch_size, 200))
    net = slim.stack(x, 
                     slim.fully_connected, 
                     [512, 256])

    # Unnormalized logits for number of classes (N) seperate K-categorical distributions
    logits_y = tf.reshape(slim.fully_connected(net, 
                                               FLAGS.num_classes*FLAGS.num_cat_dists, 
                                               activation_fn=None), 
                          [-1, FLAGS.num_cat_dists])

    q_y = tf.nn.softmax(logits_y)
    log_q_y = tf.log(q_y + 1e-20)

    return logits_y, q_y, log_q_y

def decoder(tau, logits_y):
    y = tf.reshape(gumbel_softmax(logits_y, tau, hard=False), 
                   [-1, FLAGS.num_cat_dists, FLAGS.num_classes])
    
    # Generative model p(x|y), i.e. the decoder (shape=(batch_size, 200))
    net = slim.stack(slim.flatten(y), 
                     slim.fully_connected, 
                     [256, 512])

    logits_x = slim.fully_connected(net, 
                                    784, 
                                    activation_fn=None)

    # (shape=(batch_size, 784))
    p_x = bernoulli(logits=logits_x)
    
    return p_x

def create_train_op(x,
                    lr,  
                    q_y, 
                    log_q_y, 
                    p_x):

    kl_tmp = tf.reshape(q_y * (log_q_y - tf.log(1.0 / FLAGS.num_classes)), 
                        [-1, FLAGS.num_cat_dists, FLAGS.num_classes])
    
    KL = tf.reduce_sum(kl_tmp, [1,2])
    elbo = tf.reduce_sum(p_x.log_prob(x), 1) - KL
    
    loss = tf.reduce_mean(-elbo)
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    return train_op, loss

def train():

    # Setup encoder
    # input image x (shape=(batch_size, 784))
    inputs = tf.placeholder(tf.float32, shape=[None, 784], name='inputs')
    tau = tf.placeholder(tf.float32, [], name='temperature')
    learning_rate = tf.placeholder(tf.float32, [], name='lr_value')

    # Get data i.e. MNIST
    data = input_data.read_data_sets(FLAGS.data_dir + '/MNIST', one_hot=True)
    logits_y, q_y, log_q_y = encoder(inputs)
    
    # Setup decoder
    p_x = decoder(tau, logits_y)

    train_op, loss = create_train_op(inputs, learning_rate, q_y, log_q_y, p_x)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    sess = tf.Session()
    saver = tf.train.Saver()

    sess.run(init_op)
    dat = []

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for i in tqdm(range(1, FLAGS.num_iters)):
            np_x, np_y = data.train.next_batch(FLAGS.batch_size)
            _, np_loss = sess.run([train_op, loss], {inputs: np_x, learning_rate: FLAGS.learning_rate, tau: FLAGS.init_temp})
            #_, np_loss = sess.run([train_op, loss], inputs : )

            if i % 10000 == 1:
                path = saver.save(sess, FLAGS.checkpoint_dir + '/modek.ckpt')
                print('Model saved at iteration {} in checkpoint {}'.format(i, path))
                dat.append([i, FLAGS.min_temp, np_loss])
            if i % 1000 == 1:
                FLAGS.min_temp = np.maximum(FLAGS.init_temp * np.exp(-FLAGS.anneal_rate * i), 
                                        FLAGS.min_temp)
                FLAGS.learning_rate *= 0.9
                print('Temperature updated to {}\n'.format(FLAGS.min_temp) +
                      'Learning rate updated to {}'.format(FLAGS.learning_rate))
            if i % 5000 == 1:
                print('Iteration {}\nELBO: {}\n'.format(i, -np_loss))

        #coord.request_stop()
        #coord.join(threads)
        #sess.close()

    except KeyboardInterrupt:
       print()

    finally:
        #save(saver, sess, FLAGS.log_dir, i)
        coord.request_stop()
        coord.join(threads)
        sess.close()

def main():
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.data_dir)
    tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    train()

if __name__=="__main__":
    #tf.app.run()
    main()
