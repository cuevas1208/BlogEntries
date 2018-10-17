########################################################################
#
# Functions for visualize images and data.
#
# Implemented in Python 3.5
#
########################################################################
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Manuel Cuevas
#
########################################################################
import tensorflow as tf

class Pre_processes:
    def __init__(self,num_channels, img_size_cropped):
        self.num_channels = num_channels
        self.img_size_cropped = img_size_cropped

    '''
    The following helper-functions pre-processes the input images.
    Nothing is actually calculated at this point, the function merely adds
    nodes to the computational graph for TensorFlow. This artificially inflates
    the size of the training-set by creating random variations of the original
    input images. Examples of distorted images are shown further below.

    * For training:
    randomly cropped, randomly flipped horizontally, and the hue, contrast and
    saturation is adjusted with random values.

    * For testing, the input images are cropped around the centre to fit the training model.
    '''
    def pre_process_image(self, image, training):
        if training:
            # For training, add the following to the TensorFlow graph.

            # Randomly crop the input image.
            image = tf.random_crop(image, size=[self.img_size_cropped, self.img_size_cropped, self.num_channels])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

            # Randomly adjust hue, contrast and saturation.
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

            # Some of these functions may overflow and result in pixel
            # values beyond the [0, 1] range. It is unclear from the
            # documentation of TensorFlow 0.10.0rc0 whether this is
            # intended. A simple solution is to limit the range.

            # Limit the image pixels between [0, 1] in case of overflow.
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 0.0)
        else:
            # For training, add the following to the TensorFlow graph.

            # Crop the input image around the centre so it is the same
            # size as images that are randomly cropped during training.
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           target_height=self.img_size_cropped,
                                                           target_width=self.img_size_cropped)

        return image

    def pre_process(self, images, training):
        # Use map_fn to loop over all the input images and call
        # the function pre_process_image which takes a single image as input.
        images = tf.map_fn(lambda image: self.pre_process_image(image, training), images)

        return images