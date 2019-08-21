import os
import scipy.io
import pandas as pd
import numpy as np
import cv2
import os.path as path
from datetime import datetime
from numpy import asarray
from PIL import Image
from keras.utils import plot_model, to_categorical
from sklearn.utils import shuffle
from tempfile import mkdtemp

class data_load():
    def __init__(self):
        self.db ='wiki'
        self.target_size = (224,224)

    def collectData(self):
        mat = scipy.io.loadmat('./data/{}_crop/{}.mat'.format(self.db,self.db))
        instances = mat[self.db][0][0][0].shape[1]
        columns = ['dob', 'photo_taken', 'full_path', 'gender', "name", "face_location", "face_score", "second_face_score"]
        df= pd.DataFrame(index=range(0, instances), columns=columns)
        for i in mat:
            if(i == 'wiki' or i=='imdb'):
                curr_array = mat[i][0][0]
                len_currarray = len(curr_array)
                if(i == 'imdb'):
                    len_currarray -=2
                for j in range(len_currarray):
                   df[columns[j]] = pd.DataFrame(curr_array[j][0])
        return df

    def caleAge(self, taken, dob):

        birth = datetime.fromordinal(max(int(dob) - 366, 1))
        if birth.month < 7:
            return taken - birth.year
        else:
            return taken - birth.year - 1

    # update to pixel value
    def getImagetoPixels(self, image_path):  # db = wiki or imdb

        image = cv2.imread('./data/{}_crop/{}'.format(self.db, image_path[0]), cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.target_size)

        return image.reshape(1, -1)[0]

    def dataProcessing(self, df):

        np.random.randn(12345)
        df['age'] = [self.caleAge(df['photo_taken'][i], df['dob'][i]) for i in range(len(df['dob']))]
        df = df[df['age'] > 0]
        df = df[df['age'] <= 100]
        # clean data
        # remove no face picture
        df = df[df['face_score'] != -np.inf]
        # keep only one face picture
        df = df[df['second_face_score'].isna()]
        # keep face_score >= 3
        df= df[df['face_score'] >= 3]
        # no gender
        df = df[df['gender'].isna() == False]
        # reomve unuse columns
        df = df.drop(columns=['name', 'face_score', 'second_face_score', 'face_location'])
        #add new pixel column for training
        df['pixels'] = df['full_path'].apply(self.getImagetoPixels)

        return df

    def dataOperation(self):

        datalist = ['wiki', 'imdb']
        self.db = datalist[0]
        db_wiki = self.collectData()
        db_wiki = self.dataProcessing(db_wiki)
        self.db = datalist[1]
        db_imdb = self.collectData()
        db_imdb = self.dataProcessing(db_imdb)
        print("Data collection done!")
        print("Basic data processing done!")
        #combine two dataset and shuffled
        db_combined = pd.concat([db_wiki, db_imdb])
        db_combined = shuffle(db_combined)
        print("db_combined shape {}".format(db_combined.shape))
        print("data prepared done!")

        return db_combined

    # series = pd.Series(np.random.normal(size=2000))