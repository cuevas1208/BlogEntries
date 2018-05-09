"""
Example TensorFlow script for finetuning a VGG model on your own data.
Uses tf.contrib.data module which is in release v1.2

Based on PyTorch example from Justin Johnson
https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c
https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c

Required packages: tensorflow (v1.2)
Download the weights trained on ImageNet for VGG:

```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
```

For this example we will use a tiny dataset of images from the COCO dataset:
```
wget cs231n.stanford.edu/coco-animals.zip
unzip coco-animals.zip
rm coco-animals.zip
```

The training data is stored on disk; each category has its own folder on disk
and the images for that category are stored as .jpg files in the category folder.
"""
import os
import tensorflow as tf

def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and validation get them in the same order
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i

    labels = [label_to_int[l] for l in labels]

    return filenames, labels