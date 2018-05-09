"""
# code is base on this example
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/
This code is ruunning in coco-animals dataset
"""
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
from nets import vgg
import glob
import visualize as vis
slim = tf.contrib.slim

# A place where you have downloaded a network checkpoint -- look at the previous post
model_path = '/tmp/models/animals_coco_vgg16/model.ckpt'
file_path = '/tmp/datasets/coco-animals/train/bear/*.jpg'
labels_name = os.listdir('/tmp/datasets/coco-animals/train')
labels_name.sort()

# Load the mean pixel values and the function
# that performs the subtraction
from preprocessing.vgg_preprocessing import (_mean_image_subtraction, _R_MEAN, _G_MEAN, _B_MEAN)
slim = tf.contrib.slim

def preprocess_images(x, input_width, input_height, input_depth, input_mean, input_std):
    decoded_image_as_float = tf.cast(x, dtype=tf.float32)
    expand_last_dims = tf.expand_dims(decoded_image_as_float, -1)
    if input_depth == 3:
        expand_last_dims = tf.image.grayscale_to_rgb(expand_last_dims)
    expand_first_dims = tf.expand_dims(expand_last_dims, 0)

    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resize_image = tf.image.resize_bilinear(expand_first_dims, resize_shape_as_int)

    offset_image = tf.subtract(resize_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)

    return mul_image

file_list = glob.glob(file_path)

for file_name in file_list:
    print(file_name)


    with tf.Graph().as_default():

        input_name = "file_reader"

        ''' original '''
        file_reader = tf.read_file(file_name, input_name)
        image = tf.Print(file_reader, [tf.shape(file_reader)], "file_reader")

        # image = tf.image.decode_png(image, channels=3)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.Print(image, [tf.shape(image)], "file_reader")

        # Convert image to float32 before subtracting the
        # mean pixel value
        image_float = tf.to_float(image, name='ToFloat')
        image_float = tf.Print(image_float, [tf.shape(image_float)], "image_float")
        # Subtract the mean pixel value from each pixel
        processed_image = _mean_image_subtraction(image_float, [_R_MEAN, _G_MEAN, _B_MEAN])
        processed_image = tf.Print(processed_image, [tf.shape(processed_image)], "processed_image")

        input_image = tf.expand_dims(processed_image, 0)
        input_image = tf.Print(input_image, [tf.shape(input_image)], "input_image")

        input_image = tf.image.resize_images(input_image, [480*2, 640*2])
        input_image = tf.Print(input_image, [tf.shape(input_image)], "resize_images")

        with slim.arg_scope(vgg.vgg_arg_scope()):
            # spatial_squeeze option enables to use network in a fully convolutional manner
            logits, _ = vgg.vgg_16(input_image,
                                   num_classes=len(labels_name),
                                   is_training=False,
                                   spatial_squeeze=False)

        # For each pixel we get predictions for each class. We need to pick the one with the highest probability.
        # To be more precise, these are not probabilities, because we didn't apply softmax. But if we pick a class
        # with the highest value it will be equivalent to picking the highest value after applying softmax
        # argmax returns the index with the largest value across axes of a tensor.
        pred = tf.argmax(logits, dimension=3)

        # reads the network weights from the checkpoint file that you downloaded.
        # init_fn = slim.assign_from_checkpoint_fn(model_path, slim.get_model_variables('vgg_16'))
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        new_segmentation = tf.expand_dims(pred, -1)
        image_segmentation = tf.image.resize_images(new_segmentation, [tf.shape(image)[0], tf.shape(image)[1]])
        image_segmentation = tf.squeeze(image_segmentation, [0, -1])

        with tf.Session() as sess:
            init_fn(sess)  # load the pre-trained weights
            segmentation, np_image, np_logits, image_segmentation = sess.run([pred, image, logits, image_segmentation])

    print("\n segmentation", segmentation.shape)
    print("\n np_image", np_image.shape)
    print("\n image_segmentation", image_segmentation.shape)

    vis.visualize_image_labels(np_image, segmentation, labels_name, image_segmentation)