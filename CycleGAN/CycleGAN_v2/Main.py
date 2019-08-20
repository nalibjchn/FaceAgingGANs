import matplotlib
matplotlib.use('Agg')

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
from Model import *
from data_load import *
import argparse


def strTobool(value):
    if(value.lower() == 'true'):
        return True
    else:
        return False
parser = argparse.ArgumentParser(description='CAAE')
parser.add_argument('--is_train',type=strTobool,default='True')
parser.add_argument('--pre-trained_model',type=str, default=None)
parser.add_argument('--trainA',type=str, default='TrainA_11sto20')
parser.add_argument('--trainB',type=str, default='TrainB_50sto70')
parser.add_argument('--root_path', type=str, default='../DATA/CycleGANs_Paired_TrainingSet', help='training test folder')
parser.add_argument('--path_testA', type=str,default= './test/014A18.jpg')
parser.add_argument('--path_testB', type=str, default='./test/048A54.jpg')

FLAGS = parser.parse_args()

class cycGAN():
    def __init__(self):

        self.root_path = FLAGS.root_path
        self.trainA = FLAGS.trainA
        self.trainB = FLAGS.trainB

        # hyperparameter setup
        self.lambda_A = 10.0  # cyclic loss weight A2B
        self.lambda_B = 10.0  # cyclic loss weight B2A

        self.lambda_id_A = 0.1 * self.lambda_A
        self.lambda_id_B = 0.1 * self.lambda_B

        self.lambda_D = 1.0  # weight for loss discriminator guess on sythetic image
        self.lr_D = 2e-4
        self.lr_G = 2e-4

        self.generator_iter = 1
        self.discriminator_iter = 1
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 1
        self.batch_num = 0
        self.epochs = 200
        self.save_interval = 1
        self.fake_pool_size = 50
        self.channels = 3

        self.Real_label = 1  # Use e.g. 0.9 to avoid training the discriminators to zero loss
        self.img_shape = (256, 256, 3)
        self.img_rows = 256
        self.img_columns = 256
        self.save_interval = 50

    def setup_model(self):

        # initial image pooling
        self.image_pooling = ImagePool(self.fake_pool_size)
        # optimizer
        self.opt_D = Adam(self.lr_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.lr_G, self.beta_1, self.beta_2)

        # setup discriminator model
        self.D_A = model_discriminator(self.img_shape)
        self.D_B = model_discriminator(self.img_shape)

        self.D_A.summary()

        self.loss_weights_D = [0.5]
        self.img_A = Input(shape=self.img_shape)  # real image
        self.img_B = Input(shape=self.img_shape)

        # discriminator build
        self.guess_A = self.D_A(self.img_A)
        self.guess_B = self.D_B(self.img_B)

        self.D_A = Model(inputs=self.img_A, outputs=self.guess_A, name='D_A_Model') #name for save model
        self.D_B = Model(inputs=self.img_B, outputs=self.guess_B, name='D_B_Model')

        self.D_A.compile(optimizer=self.opt_D, \
                    loss='mse', \
                    loss_weights=self.loss_weights_D, \
                    metrics=['accuracy'])

        self.D_B.compile(optimizer=self.opt_D, \
                    loss='mse', \
                    loss_weights=self.loss_weights_D, \
                    metrics=['accuracy'])

        # for generator model, we do not train discriminator
        self.D_A_static = Network(inputs=self.img_A, outputs=self.guess_A, name='D_A_static_model')
        self.D_B_static = Network(inputs=self.img_B, outputs=self.guess_B, name='D_B_static_model')

        #generator setup
        self.G_A2B = model_generator(self.channels, self.img_shape, name='G_A2B_model')
        self.G_B2A = model_generator(self.channels, self.img_shape, name='G_B2A_model')

        self.G_A2B.summary()

        # # import image
        # self.img_A = Input(shape=self.img_shape)
        # self.img_B = Input(shape=self.img_shape)

        # generate fake images, transfer image from A to B
        self.fake_B = self.G_A2B(self.img_A)
        self.fake_A = self.G_B2A(self.img_B)

        # reconstruction, transfer to original image from fake image
        self.reconstor_A = self.G_B2A(self.fake_B)
        self.reconstor_B = self.G_A2B(self.fake_A)

        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Discriminators determines validity of translated images
        self.valid_A = self.D_A_static(self.fake_A)
        self.valid_B = self.D_A_static(self.fake_B)

        # identity learning
        self.identity_A = self.G_B2A(self.img_A)
        self.identity_B = self.G_A2B(self.img_B)

        # combined two models and compile
        # ombined model trains generators to fool discriminators

        self.combined_model = Model(inputs=[self.img_A, self.img_B], \
                                    outputs=[self.valid_A, self.valid_B, \
                                    self.reconstor_A, self.reconstor_B, \
                                    self.identity_A, self.identity_B],
                                    name='Combined_G_model')

        self.combined_model.compile(loss=['mse', 'mse', \
                                           'mae', 'mae', \
                                           'mae', 'mae'],\
                                    loss_weights=[self.lambda_D, self.lambda_D, \
                                                  self.lambda_A, self.lambda_B, \
                                                  self.lambda_id_A, self.lambda_id_B],\
                                    optimizer=self.opt_G)

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        # Create a session with the above options specified.
        kerasbackend.tensorflow_backend.set_session(tf.Session(config=self.config))

        # patchGAN
        # output shape of D(PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

    def saveimage(self, batch_index, epoch_index, path_testA = FLAGS.path_testA, path_testB=FLAGS.path_testB):

        os.makedirs('images', exist_ok=True)
        rows, columns = 2, 3
        img_A = imread(path_testA, pilmode='RGB').astype(np.float32)
        img_B = imread(path_testB,pilmode='RGB').astype(np.float32)

        img_A = resize(img_A, (256, 256))
        img_B = resize(img_B, (256, 256))
        #   normalization
        imgs_A, imgs_B = [], []
        imgs_A.append(img_A)
        imgs_A = np.array(imgs_A) / 127.5 - 1

        imgs_B.append(img_B)
        imgs_B = np.array(imgs_B) / 127.5 - 1

        # transform other domain
        fake_B = self.G_A2B.predict(imgs_A)
        fake_A = self.G_B2A.predict(imgs_B)

        # recontract to orginal domain
        recon_A = self.G_B2A.predict(fake_B)
        recon_B = self.G_A2B.predict(fake_A)

        imgs = np.concatenate([imgs_A, fake_B, recon_A, imgs_B, fake_A, recon_B])

        # rescale image 0-1
        imgs = 0.5 * imgs + 0.5
        # print(batch_index)
        titles = ['original', 'transform', 'goback']
        fig, axs = plt.subplots(rows, columns)
        count = 0
        for i in range(rows):
            for j in range(columns):
                axs[i, j].imshow(imgs[count])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                count += 1
        if FLAGS.is_train:
            plt.savefig("images/%d_%d.png" % (epoch_index, batch_index))
        else:
            plt.savefig("images/test_{}.png".format(batch_index))
        plt.close()

    def train(self):

        # train once
        # Training discriminators one-hot vector
        valid = np.ones((self.batch_size,) + self.disc_patch)
        fake = np.zeros((self.batch_size,) + self.disc_patch)

        for epoch in range(self.epochs):

            # update learning rate (decay) for each epoch
            self.lr_D, self.lr_G = update_lr(epoch, self.lr_D, self.lr_G)
            kerasbackend.set_value(self.D_A.optimizer.lr, self.lr_D)
            kerasbackend.set_value(self.D_B.optimizer.lr, self.lr_D)
            kerasbackend.set_value(self.combined_model.optimizer.lr, self.lr_G)

            for batch_index, (batch_num, imgs_A, imgs_B) in enumerate(loaddata_batch(self.batch_size, self.root_path, self.trainA,self.trainB)):

                fake_B_tmp = self.G_A2B.predict(imgs_A)
                fake_A_tmp = self.G_B2A.predict(imgs_B)

                fake_B = self.image_pooling.fake_image_pooling(fake_B_tmp)
                fake_A = self.image_pooling.fake_image_pooling(fake_A_tmp)

                dA_loss_real = self.D_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.D_A.train_on_batch(fake_A, fake)

                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.D_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.D_B.train_on_batch(fake_B, fake)

                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # training generation
                g_loss = self.combined_model.train_on_batch([imgs_A, imgs_B], \
                                                       [valid, valid, \
                                                        imgs_A, imgs_B, \
                                                        imgs_A, imgs_B])

                print("[epoch_index: %d/%d][batch_index:%d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] [time:%s]" \
                    % (epoch, self.epochs, \
                       batch_index, batch_num, \
                       d_loss[0], 100 * d_loss[1], \
                       g_loss[0], \
                       np.mean(g_loss[1:3]), \
                       np.mean(g_loss[3:5]), \
                       np.mean(g_loss[5:6]), \
                       datetime.datetime.now()))
                if batch_index % self.save_interval == 0:
                    print(
                        "<<=========save image + test image + original image + reconstruct image + return original image============>>")
                    self.saveimage(batch_index, epoch, FLAGS.path_testA, FLAGS.path_testB) #save and test

            if (epoch % 50 == 0):
                save_model(epoch, self.D_A)
                save_model(epoch, self.D_B)
                save_model(epoch, self.combined_model)
                save_model(epoch, self.G_A2B)
                save_model(epoch, self.G_B2A)

    def test(self, modelname):
       #load weights
       self.combined_model.load_weights('weight_combined_G_model.h5')
       self.D_A.load_weights('weight_D_A_Model.h5')
       self.D_B.load_weights('weight_D_B_Model.h5')
       self.G_B2A.load_weights('weight_G_B2A_model.h5')
       self.G_A2B.load_weights('weight_G_A2B_model.h5')

       #batch index and epoch = -1 as test
       self.saveimage(-1,-1)

def main():
    model = cycGAN()
    model.setup_model()

    if(FLAGS.is_train):
        model.train()
    else:
        model.test()

if __name__=='__main__':
    main()

