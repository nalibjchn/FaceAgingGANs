import scipy.io
import os
import pandas as pd
import numpy as np
import cv2
from datetime import datetime

from numpy import asarray
from PIL import Image

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
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from tempfile import mkdtemp
import os.path as path
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras_vggface.vggface import VGGFace


def cale_age(taken, dob):
  
  birth = datetime.fromordinal(max(int(dob) - 366,1))
  
  if birth.month < 7:
    return taken - birth.year
  else:
    return taken - birth.year - 1

def getImagetoPixels_wiki(image_path, db='wiki'): #db = wiki or imdb
  image = cv2.imread('./data/{}_crop/{}'.format(db, image_path[0]), cv2.IMREAD_COLOR)
  image = cv2.resize(image,target_size)
#   print(db)
  return image.reshape(1,-1)[0]

def getImagetoPixels_imdb(image_path, db='imdb'): #db = wiki or imdb
  image = cv2.imread('./data/{}_crop/{}'.format(db, image_path[0]), cv2.IMREAD_COLOR)
  image = cv2.resize(image,target_size)
  return image.reshape(1,-1)[0]


if __name__ == '__main__':

    mat = scipy.io.loadmat('./data/wiki_crop/wiki.mat')
    instances = mat['wiki'][0][0][0].shape[1]
    columns = ['dob', 'photo_taken', 'full_path', 'gender', "name", "face_location", "face_score", "second_face_score"]
    df_wiki = pd.DataFrame(index=range(0, instances), columns=columns)
    for i in mat:
      if i == "wiki":
        curr_array = mat[i][0][0]
        for j in range(len(curr_array)):
          df_wiki[columns[j]] = pd.DataFrame(curr_array[j][0])
    
    mat = scipy.io.loadmat('./data/imdb_crop/imdb.mat')
    instances_imdb = mat['imdb'][0][0][0].shape[1]
    df_imdb = pd.DataFrame(index=range(0, instances_imdb), columns=columns)
    for i in mat:
      if i == "imdb":
        curr_array = mat[i][0][0]
        print(len(curr_array))
        for j in range(0, len(curr_array)-2): #imdb has 10 columns
          df_imdb[columns[j]] = pd.DataFrame(curr_array[j][0])

    df_wiki['age'] = [cale_age(df_wiki['photo_taken'][i], df_wiki['dob'][i]) for i in range(len(df_wiki['dob']))]
    df_imdb['age'] = [cale_age(df_imdb['photo_taken'][i], df_imdb['dob'][i]) for i in range(len(df_imdb['dob']))]

    #clean data
    #remove no face picture
    df_wiki = df_wiki[df_wiki['face_score'] != -np.inf]
    #remove more faces in one picuture
    df_wiki = df_wiki[df_wiki['second_face_score'].isna()]
    #threshold more than 3
    df_wiki = df_wiki[df_wiki['face_score'] >= 3]
    #remove no gender
    df_wiki = df_wiki[df_wiki['gender'].isna()==False]
    #reomve unuse columns
    df_wiki = df_wiki.drop(columns=['name', 'face_score', 'second_face_score', 'face_location'])

    #clean data
    #remove no face picture
    df_imdb = df_imdb[df_imdb['face_score'] != -np.inf]
    #remove more faces in one picuture
    df_imdb = df_imdb[df_imdb['second_face_score'].isna()]
    #threshold more than 3
    df_imdb = df_imdb[df_imdb['face_score'] >= 3]
    #remove no gender
    df_imdb = df_imdb[df_imdb['gender'].isna()==False]
    #reomve unuse columns
    df_imdb = df_imdb.drop(columns=['name', 'face_score', 'second_face_score', 'face_location'])

    df_wiki = df_wiki[df_wiki['age']>0]
    df_wiki = df_wiki[df_wiki['age']<=100]
    df_imdb = df_imdb[df_imdb['age']>0]
    df_imdb = df_imdb[df_imdb['age']<=100]

    # df = df[df['age']>0]
    # df = df[df['age']<=100]

    series = pd.Series(np.random.normal(size=2000))
    #check gender distribution 
    target_size = (224,224)
    #update to pixel value
    df_wiki['pixels'] = df_wiki['full_path'].apply(getImagetoPixels_wiki)

    df_imdb['pixels'] = df_imdb['full_path'].apply(getImagetoPixels_imdb)
     
    df = df_imdb
    #df = df_wiki.append(df_imdb, sort=False)
    df = shuffle(df) #shuffle combined dataset
    
    #perpare training and test data with labels.
    num_gender_classes = 2

    target_classes = to_categorical(df['age'].values, num_gender_classes)

    image_features = []
    for i in range(0, df.shape[0]):
      image_features.append(df['pixels'].values[i])

    #fp_image_features_array.flush()
    image_features = np.array(image_features, dtype='float32')
    image_features = image_features.reshape(image_features.shape[0], 224,224,3)
       
    image_features = image_features/255
    
    
    train_x, test_x, train_y, test_y = train_test_split(image_features, target_classes, test_size=0.3)
    vgg_model = VGGFace(include_top=False, model='vgg16', weights='vggface', input_shape=(224, 224, 3))

    #add top layers
    last_layer  = vgg_model.get_layer('pool5').output
    for layer in vgg_model.layers:
      layer.trainable = False

    x = Conv2D(4096, (7, 7), activation='relu')(last_layer)
    # x = Flatten(name='flatten')(last_layer)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(2, (1,1),name="prediction")(x) #gender
    # x = Dense(4496, activation='relu', name='fc6')(x)
    x = Flatten(name='flatten')(x)
    output = Activation('softmax')(x)

    facegender_vgg_model = Model(vgg_model.input, output)

    facegender_vgg_model.compile(loss='categorical_crossentropy', \
                              optimizer=Adam(), \
                              metrics=['accuracy'])

    for layer in facegender_vgg_model.layers:
        print(layer, layer.trainable)

    checkpointer = ModelCheckpoint(filepath='gender_model.h5', \
                                   monitor= 'val_loss', \
                                   verbose=1, \
                                   save_best_only=True,\
                                   mode = 'auto'
                                   )
    loss_track = TensorBoard(log_dir='face_gender_estimator_logs', histogram_freq=0, \
                            batch_size=128, \
                            write_graph=True,\
                            write_grads=False, \
                            write_images=False, \
                            embeddings_freq=0, \
                            embeddings_layer_names=None, \
                            embeddings_metadata=None,\
                            embeddings_data=None,\
                            update_freq='epoch')

    scores = []
    #epochs = 501
    #batch_size = 256
    score = facegender_vgg_model.fit(train_x, train_y, \
                              batch_size=128,\
                              epochs=300,\
                              validation_data=(test_x, test_y), \
                              callbacks=[checkpointer, loss_track])
    facegender_vgg_model = load_model("gender_model.h5")   
    facegender_vgg_model.save_weights('gender_model_weights.h5')
    print("training done!!")

