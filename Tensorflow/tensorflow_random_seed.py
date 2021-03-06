# -*- coding: utf-8 -*-
"""Tensorflow random seed
"""

import tensorflow as tf
import numpy as np

mu = 0
sigma = 0.3

# variables with/without function seed
fc1_W = tf.Variable(tf.truncated_normal(shape=(1, 2), mean=mu, stddev=sigma, seed=1))
fc2_W = tf.Variable(tf.truncated_normal(shape=(1, 2), mean=mu, stddev=sigma))

# initialize session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('\n variables with/without function seed')
    print('   fc1_W                   fc2_W')
    print(np.c_[fc1_W.eval(sess), fc2_W.eval(sess)],                   'round 1.0')

    sess.run(tf.global_variables_initializer())
    print(np.c_[fc1_W.eval(sess), fc2_W.eval(sess)],                   'round 1.1 running it again in the same session')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(np.c_[fc1_W.eval(sess), fc2_W.eval(sess)],                   'round 1.2 running it in a new session')

tf.reset_default_graph()
# using global seed
tf.set_random_seed(1)
fc1_W = tf.Variable(tf.truncated_normal(shape=(1, 2), mean=mu, stddev=sigma))
fc2_W = tf.Variable(tf.truncated_normal(shape=(1, 2), mean=mu, stddev=sigma))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('\n variables with global seed ')
    print('   fc1_W                   fc2_W')
    print(np.c_[fc1_W.eval(sess), fc2_W.eval(sess)],                   'round 2.0')

    # running function within the same session
    sess.run(tf.global_variables_initializer())
    print(np.c_[fc1_W.eval(sess), fc2_W.eval(sess)],                   'round 2.1 running it again in the same session')

# new session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(np.c_[fc1_W.eval(sess), fc2_W.eval(sess)],                   'round 2.2 running it in a new session')

# change the name for the variables, and initialize new session using global seed 1
fc1_W = tf.Variable(tf.truncated_normal(shape=(1, 2), mean=mu, stddev=sigma))
fc2_W = tf.Variable(tf.truncated_normal(shape=(1, 2), mean=mu, stddev=sigma))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(np.c_[fc1_W.eval(sess), fc2_W.eval(sess)],                   'round 2.4 initialize variables in a new session')

# change the name for the variables, restart and initialize new session using global seed 1
tf.reset_default_graph()
tf.set_random_seed(1)
fc2_W = tf.Variable(tf.truncated_normal(shape=(1, 2), mean=mu, stddev=sigma))
fc3_W = tf.Variable(tf.truncated_normal(shape=(1, 2), mean=mu, stddev=sigma))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('   fc2_W                   fc3_W')
    print(np.c_[fc2_W.eval(sess), fc3_W.eval(sess)],                   'round 2.5 change variables name')


