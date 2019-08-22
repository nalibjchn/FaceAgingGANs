from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.layers import Layer, Input, Dropout, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, Sequential
from keras.utils import plot_model, to_categorical
from keras.engine.topology import Network #for untrainable discrimator model but weight still is updated
from keras.callbacks import ModelCheckpoint

"""## Involving VGG FACE
https://github.com/rcmalli/keras-vggface.git
"""
# pip install git+https://github.com/rcmalli/keras-vggface.git
from keras_vggface.vggface import VGGFace


def ageModel(input_shape=(224, 224, 3)):

    vgg_model = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=input_shape)
    print("Base VGG model summary.")
    vgg_model.summary()

    # add top layers
    last_layer = vgg_model.get_layer('pool5').output
    for layer in vgg_model.layers:
       layer.trainable = False

    mid_layer = Conv2D(4096, (7, 7), activation='relu')(last_layer)
    mid_layer = Dropout(0.5)(mid_layer)
    mid_layer = Conv2D(4096, (1, 1), activation='relu')(mid_layer)
    mid_layer = Dropout(0.5)(mid_layer) #prevent overfitting issue
    mid_layer = Conv2D(101, (1, 1), name="prediction")(mid_layer)
    mid_layer = Flatten(name='flatten')(mid_layer)
    output = Activation('softmax')(mid_layer)

    faceage_vgg_model = Model(vgg_model.input, output)

    print("Face age model summary.")
    faceage_vgg_model.summary()
    return faceage_vgg_model

