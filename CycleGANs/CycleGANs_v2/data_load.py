from keras.layers import Layer, Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network

import numpy as np
import random
import datetime
import time

import sys
import os
import cv2
import matplotlib.pyplot as plt
import keras.backend as kerasbackend
import tensorflow as tf

from imageio import imread, imsave
from skimage.transform import resize
from glob import glob

def convert_image_array(batch_list, path):
    img_array = []

    for img in batch_list:
        img = imread(img, pilmode='RGB').astype(np.float32)
        img = resize(img, (256, 256))
        img_array.append(img)

    img_array = np.array(img_array) / 127.5 - 1
    return img_array

def loaddata_batch(batch_size,root_path, train_A, train_B):
    # data_type = 'train' if not is_testing else 'test'

    path_A = glob(os.path.join(root_path, train_A, '*.jpg'))
    path_B = glob(os.path.join(root_path, train_B, '*.jpg'))

    batch_num = int(min(len(path_A), len(path_B)) / batch_size)
    sample_num = batch_num * batch_size

    path_A = np.random.choice(path_A, sample_num, replace=False)
    path_B = np.random.choice(path_B, sample_num, replace=False)

    for i in range(batch_num):
        batch_A = path_A[i * batch_size: (i + 1) * batch_size]
        batch_B = path_B[i * batch_size: (i + 1) * batch_size]
        imgs_A = convert_image_array(batch_A, path_A[0])
        imgs_B = convert_image_array(batch_B, path_B[0])

        yield batch_num, imgs_A, imgs_B

