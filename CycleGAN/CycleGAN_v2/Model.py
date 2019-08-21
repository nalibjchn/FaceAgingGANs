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


#discriminator layer
def d_layer(inputlayer, filter_dim=64, use_normal=True):
  inputlayer = Conv2D(filters=filter_dim, kernel_size=4, strides=2, padding='same')(inputlayer)
  inputlayer = LeakyReLU(alpha=0.2)(inputlayer)
  if use_normal:
      inputlayer = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(inputlayer, training=True)
  return inputlayer


# generator layer
def g_firstLayer(inputlayer, filter_dim=32):
    inputlayer = Conv2D(filters=filter_dim, kernel_size=7, strides=1, padding='valid')(inputlayer)
    inputlayer = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(inputlayer, training=True)

    inputlayer = Activation('relu')(inputlayer)

    return inputlayer


def g_layer(inputlayer, filter_dim=64):
    inputlayer = Conv2D(filters=filter_dim, kernel_size=3, strides=2, padding='same')(inputlayer)
    inputlayer = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(inputlayer, training=True)

    inputlayer = Activation('relu')(inputlayer)

    return inputlayer


def resnet_layer(inputlayer):
    filter_size = int(inputlayer.shape[-1])

    # first layer reflection
    layer = ReflectionPadding2D((1, 1))(inputlayer)
    layer = Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='valid')(layer)
    layer = Activation('relu')(layer)

    # second layer
    layer = ReflectionPadding2D((1, 1))(layer)
    layer = Conv2D(filters=filter_size, kernel_size=3, strides=1, padding='valid')(layer)

    layer = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(layer, training=True)

    resnetlayer = add([layer, inputlayer])

    #   x = Activation('relu')(x) #from paper

    return resnetlayer


def g_upsamplinglayer(inputlayer, filter_dim=64):
    #   use_resize_convolution
    #   inputlayer = UpSampling2D(size=(2,2))(inputlayer)
    #   inputlayer = ReflectionPadding2D((1,1))(inputlayer)
    #   inputlayer = Conv2D(filters=filter_dim, kernel_size=3, strides=1, padding='valid')(inputlayer)
    inputlayer = Conv2DTranspose(filters=filter_dim, kernel_size=3, strides=2, padding='same')(inputlayer)
    inputlayer = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(inputlayer, training=True)
    inputlayer = Activation('relu')(inputlayer)

    return inputlayer


def model_discriminator(img_shape=(256, 256, 3), name=None):
    input_img = Input(shape=img_shape)
    # 1st layer no normalization
    layer = d_layer(input_img, 64, use_normal=False)
    # 2nd layer with normalizaiont
    layer = d_layer(layer, 64 * 2, use_normal=True)
    # 3rd layer
    layer = d_layer(layer, 64 * 2 * 2, use_normal=True)
    # 4th layer
    layer = d_layer(layer, 64 * 2 * 2 * 2, use_normal=True)
    # patchgan
    validity = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(layer)

    return Model(inputs=input_img, outputs=validity, name=name)


# build_generator with 9 resnet
def model_generator(channels=3, img_shape=(256, 256, 3), name=None):
    # input
    input_img = Input(shape=img_shape)

    # layer 1
    output_img = ReflectionPadding2D((3, 3))(input_img)

    output_img = g_firstLayer(output_img, 32)

    # layer 2
    output_img = g_layer(output_img, 32 * 2)

    # layer 3
    output_img = g_layer(output_img, 32 * 2 * 2)

    # layer 4~12 9 resnet
    for _ in range(4, 13):
        output_img = resnet_layer(output_img)

    # layer 13 upsamling
    output_img = g_upsamplinglayer(output_img, 32 * 2)

    # layer 14 upsamiling
    output_img = g_upsamplinglayer(output_img, 32)

    output_img = ReflectionPadding2D((3, 3))(output_img)
    output_img = Conv2D(filters=channels, kernel_size=7, strides=1)(output_img)

    output_img = Activation('tanh')(output_img)

    return Model(inputs=input_img, outputs=output_img, name=name)


def save_model(epoch, model):
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_path = os.path.join('save_model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model.save("save_model/{}_{}.h5".format(model.name, epoch))
    model.save_weights("save_model/weight_{}_{}.h5".format(model.name, epoch))
    print("model and the weights:{} has been saved at {}!".format(model.name, time))

def update_lr(epoch, lr_D=2e-4, lr_G=2e-4):
  if epoch < 100:
    lr_D = 2e-4
    lr_G = 2e-4
  else:
    lr_D = 2e-4 - 2e-4 * (epoch - 100) / 100
    lr_G = 2e-4 - 2e-4 * (epoch - 100) / 100
    if (lr_D < 0 or lr_G < 0):
      lr_D = 0
      lr_G = 0
  return lr_D, lr_G

#=========================Tool Class===================================================
# create new images from generated images
# reference https://github.com/simontomaskarlsson/CycleGAN-Keras.git
class ImagePool():

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def fake_image_pooling(self, image):

        if self.pool_size == 0:
            return image
        update_image = []

        if len(image.shape) == 3:
            image = image[np.newaxis, :, :, :]  # [1, 256, 256, 3]

        if self.num_imgs < self.pool_size:
            self.num_imgs += 1

            if len(self.images) == 0:
                self.images = image
            else:
                self.images = np.vstack((self.images, image))

            update_image = image

        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self.pool_size - 1)
                temp = self.images[random_id, :, :, :]
                temp = temp[np.newaxis, :, :, :]
                self.images[random_id, :, :, :] = image[0, :, :, :]
                update_image = temp
            else:
                update_image = image

        return update_image
# reference class
# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')



