from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import manage_images
import numpy as np
IMG_CLASSES = ['cat', 'dog']

DATA_DIR = 'data/'
TRAIN_DATA_PATH = 'data/train/'
TEST_DATA_PATH = 'data/test/'

IMG_HEIGHT = int(224)
IMG_WIDTH = int(224)
IMG_CHANNELS = 3
NUM_FILES_DATASET = 25000 #数据集大小 25000; 训练集大小 17500; 验证集大小 7500
VALIDATION_SET_FRACTION = 0.3 #validation size properbility
NUM_TRAIN_EXAMPLES = int((1 - VALIDATION_SET_FRACTION) * NUM_FILES_DATASET)
NUM_VALIDATION_EXAMPLES = int((VALIDATION_SET_FRACTION) * NUM_FILES_DATASET)
NUM_KAGGLE_TEST = 12500

def pre_processing(data_set='train',batch_size=32):
    if data_set is 'train':
        images, labels = manage_images.read_images(TRAIN_DATA_PATH, IMG_CLASSES,
                                                   IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        # Random sample
        validation_images = []
        validation_labels = []
        train_images = []
        train_labels = []
        validation_size = int(VALIDATION_SET_FRACTION * len(images))
        idx = np.random.permutation(len(images))
        for i in idx:
            if i < validation_size:
                validation_images.append(images[i])
                validation_labels.append(labels[i])
            else:
                train_images.append(images[i])
                train_labels.append(labels[i])

        train_batch_num = NUM_TRAIN_EXAMPLES // batch_size
        validation_batch_num = NUM_VALIDATION_EXAMPLES // batch_size
        pointer = 0
        batch_train_images = []
        batch_train_labels = []
        batch_validation_images = []
        batch_validation_labels = []
        for _ in range(train_batch_num):
            batch_train_images.append(train_images[pointer:pointer + batch_size])
            batch_train_labels.append(train_labels[pointer:pointer + batch_size])
            pointer = pointer + batch_size
        pointer = 0
        for _ in range(validation_batch_num):
            batch_validation_images.append(validation_images[pointer:pointer + batch_size])
            batch_validation_labels.append(validation_labels[pointer:pointer + batch_size])
            pointer = pointer + batch_size

        batch_validation_set = {'images': batch_validation_images, 'labels': batch_validation_labels}
        batch_train_set = {'images': batch_train_images, 'labels': batch_train_labels}
        image_num = {'train': NUM_TRAIN_EXAMPLES, 'validation': NUM_VALIDATION_EXAMPLES}
        return batch_train_set, batch_validation_set, image_num
    elif data_set is 'test':
        images = manage_images.read_images_kaggle_result(TEST_DATA_PATH, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

        batch_test_images = []
        batch_num = NUM_KAGGLE_TEST // batch_size

        pointer = 0
        for _ in range(batch_num):
            batch_test_images.append(images[pointer:pointer + batch_size])
            pointer = pointer + batch_size

        batch_test_set = {'images': batch_test_images}

        return batch_test_set