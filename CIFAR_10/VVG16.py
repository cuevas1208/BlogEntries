'''
example code comes from:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
'''

import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import cifar10
import os

# The CIFAR-10 data-set is about 163 MB and will be downloaded automatically
if os.path.exists("./data/CIFAR-10/"):
    cifar10.data_path = "./data/CIFAR-10/"
else:
    cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

# init slim, get vgg net
slim = tf.contrib.slim
vgg = nets.vgg

# create log dir
train_log_dir = "./vgg_log"
if not tf.gfile.Exists(train_log_dir):
  tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
  # Set up the data loading:
  images, labels = images_train.astype('float32'), cls_train

  # Define the model:
  predictions = vgg.vgg_16(images, is_training=True)

  # Specify the loss function:
  slim.losses.softmax_cross_entropy(predictions, labels)

  total_loss = slim.losses.get_total_loss()
  tf.summary.scalar('losses/total_loss', total_loss)

  # Specify the optimization scheme:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

  # create_train_op that ensures that when we evaluate it to get the loss,
  # the update_ops are done and the gradient updates are computed.
  train_tensor = slim.learning.create_train_op(total_loss, optimizer)

  # Actually runs training.
  slim.learning.train(train_tensor, train_log_dir)