
## Overview
This is a practical example on using Tensorflow for finetuning a VGG model with your own data.

For more information about the code click in the link bellow: 
http://www.guidetomlandai.com/tutorials/computer_vision/How-to-fine-tune-a-deep-neural-network-using-Tensorflow/


## Setup
Download project files by clicking ->> [Download LINK] or clone repository

## Pre-requirements 
Tensorflow example uses tf.contrib.data module which is in release v1.2

>Note: the location of where I placed packages bellow are aligned with the current project, feel free to modified it as need it 

Download the weights trained on ImageNet for VGG:
```
    cd /tmp/checkpoints/
    wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
    tar -xvf vgg_16_2016_08_28.tar.gz
    rm vgg_16_2016_08_28.tar.gz
```

For this example we will use a tiny dataset of images from the COCO dataset.
We have chosen eight types of animals (bear, bird, cat, dog, giraffe, horse,
sheep, and zebra); for each of these categories we have selected 100 training
images and 25 validation images from the COCO dataset. You can download and
unpack the data (176 MB) by running:

```python
    cd /tmp/datasets
    wget cs231n.stanford.edu/coco-animals.zip
    unzip coco-animals.zip
    rm coco-animals.zip

```


# RUN CODE

## 1. Fine tune your model
Before you [finetuning_vgg16.py], modify the following lines as shown bellow: 
line 34 and 35 with the location of your training and validation directories
Line 36 with the pre-trained checkpoint
Line 45 with the output directory 

```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/tmp/datasets/coco-animals/train')
    parser.add_argument('--val_dir', default='/tmp/datasets/coco-animals/val')
    parser.add_argument('--model_path', default='/tmp/checkpoints/vgg_16.ckpt', type=str)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_epochs1', default=10, type=int)
    parser.add_argument('--num_epochs2', default=10, type=int)
    parser.add_argument('--learning_rate1', default=1e-3, type=float)
    parser.add_argument('--learning_rate2', default=1e-4, type=float)
    parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--output_checkpoint', default='/tmp/models/animals_coco_vgg16/model.ckpt', type=str)
```
run:
```python
python finetuning_vgg16.py
```

## 2. Run model
```python
python FCN_vgg16.py
```

## Reference 

[1]:Fine tuning reference code
[2]:Based on PyTorch example from Justin Johnson

[Download LINK]:https://minhaskamal.github.io/DownGit/#/home?url=https:%2F%2Fgithub.com%2Fcuevas1208%2FML_Notes_and_Research%2Ftree%2Fmaster%2Fcomputer_vision%2FFCN_32_vgg16
[1]:https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c
[2]:https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c
