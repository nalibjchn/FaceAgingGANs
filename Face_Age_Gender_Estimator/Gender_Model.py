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

class genderModel():

    def __init__(self,
                 input_shape=(224, 224, 3),
                 model_type='vgg16'):

        self.input_shape = input_shape
        self.model_type = model_type
        self.vggface_model = VGGFace(include_top=False, model=self.model_type, weights='vggface',\
                            input_shape=self.input_shape)

        print("Base VGG model summary.")
        self.vggface_model.summary()

    def vggface_vgg16_gender(self):
        # add top layers, fix other layers
        last_layer = self.vggface_model.get_layer('pool5').output
        for layer in self.vggface_model.layers:
           layer.trainable = False

        mid_layer = Conv2D(4096, (7, 7), activation='relu')(last_layer)
        mid_layer = Dropout(0.5)(mid_layer)
        mid_layer = Conv2D(4096, (1, 1), activation='relu')(mid_layer)
        mid_layer = Dropout(0.5)(mid_layer) #prevent overfitting issue
        mid_layer = Conv2D(2, (1, 1), name="prediction")(mid_layer)
        mid_layer = Flatten(name='flatten')(mid_layer)
        output = Activation('softmax')(mid_layer)

        self.facegender_vgg16_model = Model(self.model_type.input, output)

        print("face gender (VGG_16) model summary.")
        self.facegender_vgg16_model.summary()

    def vggface_resnet50_gender(self):

       # add top layers
        last_layer = self.vggface_model.get_layer('avg_pool').output
        for layer in self.vggface_model.layers:
            layer.trainable = False
        x = Flatten(name='customer_flatten')(last_layer)
        output = Dense(2, activation='softmax', name='perdiction')(x)
        self.facegender_resnet50_model = Model(self.vggface_model.input, output)
        print("face gender (resnet_50) model summary.")
        self.facegender_resnet50_model.summary()
