from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, Sequential, load_model
from keras.utils import plot_model, to_categorical
from keras.engine.topology import Network  # for untrainable discrimator model but weight still is updated
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import argparse

# customer lib
from Data_Load import *
from Age_Model import *
from Gender_Model import *

def getarg():

    parser = argparse.ArgumentParser(description="get basic parameters for training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_model", type=str, default='age',
                        help="age or gender for training")

    args = parser.parse_args()
    return args



class estimator():
    def __init__(self,
                 input_shape=(224, 224, 3),
                 testsize=0.3,
                 epochs=250,
                 batch_size=80):
        self.input_shape = input_shape
        self.row = 224
        self.col = 224
        self.channel = 3
        self.testsize = testsize
        self.epochs = epochs
        self.batch_size = batch_size

    def normalizationData(self, image_array):
        image_array = image_array / 255
        return image_array

    def trainingDataPrepared(self, df):
        # perpare training and test data with labels.
        num_age_classes = 101  # age from 0 to 100
        target_classes = to_categorical(df['age'].values, num_age_classes)
        image_features = []
        for i in range(0, df.shape[0]):
            image_features.append(df['pixels'].values[i])

        # save memory
        filename_array = path.join(mkdtemp(), 'image_features_array.dat')
        fp_image_features_array = np.memmap(filename_array, dtype='float32', mode='w+', \
                                            shape=(len(image_features), self.row * self.col * self.channel))

        fp_image_features_array[:] = np.array(image_features)

        filename = path.join(mkdtemp(), 'imageFeaturefile.dat')
        fp = np.memmap(filename, dtype='float32', mode='w+', shape=(fp_image_features_array.shape[0], self.row, self.col, self.channel))
        fp[:] = fp_image_features_array[:].reshape(fp_image_features_array.shape[0], self.row, self.col, self.channel)
        fp[:] = self.normalizationData(fp)

        #save RAM if the RAM is limited on Google Colab
        # train_x_mp = path.join(mkdtemp(), 'train_x.dat')
        # train_x = np.memmap(train_x_mp, dtype='float32', mode='w+', shape=(round(fp.shape[0] * (1-self.testsize)), self.row, self.col, self.channel))
        #
        # train_y_mp = path.join(mkdtemp(), 'train_y.dat')
        # train_y = np.memmap(train_y_mp, dtype='float32', mode='w+', shape=(round(fp.shape[0] * (1-self.testsize)), num_age_classes))
        #
        # test_x_mp = path.join(mkdtemp(), 'test_x.dat')
        # test_x = np.memmap(test_x_mp, dtype='float32', mode='w+', shape=(round(fp.shape[0] * self.testsize), self.row, self.col, self.channel))
        #
        # test_y_mp = path.join(mkdtemp(), 'test_y.dat')
        # test_y = np.memmap(test_y_mp, dtype='float32', mode='w+', shape=(round(fp.shape[0] * self.testsize), num_age_classes))

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(fp, target_classes, test_size=self.testsize)

        return self.train_x, self.test_x, self.train_y, self.test_y

    def train_agemodel(self):

        # initial face model
        self.faceage_vgg_model = ageModel(self.input_shape)
        self.faceage_vgg_model.compile(loss='categorical_crossentropy', \
                                  optimizer=Adam(lr=1e-3), \
                                  metrics=['accuracy'])

        for layer in self.faceage_vgg_model.layers:
            print(layer, layer.trainable)

        checkpointer = ModelCheckpoint(filepath='face_age_model.hd5f', \
                                       monitor='val_loss', \
                                       verbose=1, \
                                       save_best_only=True, \
                                       mode='auto'
                                       )

        loss_track = TensorBoard(log_dir='./face_age_estimator_logs', histogram_freq=0, \
                                 batch_size=self.batch_size, \
                                 write_graph=True, \
                                 write_grads=False, \
                                 write_images=False, \
                                 embeddings_freq=0, \
                                 embeddings_layer_names=None, \
                                 embeddings_metadata=None, \
                                 embeddings_data=None, \
                                 update_freq='epoch')
        # for i in range(0, epochs):
        #     print("epoch ", i)
        #     batch_train = np.random.choice(self.train_x.shape[0], size=batch_size)
        #     score = faceage_vgg_model.fit(self.train_x[batch_train], self.train_y[batch_train], \
        #                                   epochs=1, \
        #                                   validation_data=(self.test_x, self.test_y), \
        #                                   callbacks=[checkpointer,loss_track])

        score = self.faceage_vgg_model.fit(self.train_x, self.train_y, \
                                      batch_size=self.batch_size, \
                                      epochs=self.epochs, \
                                      validation_data=(self.test_x, self.test_y), \
                                      callbacks=[checkpointer, loss_track])

        self.faceage_vgg_model = load_model("face_age_model.h5") #optional: load the best model before test
        self.faceage_vgg_model.save_weights('face_age_model_weights.h5')
        print("Age Training is done")

    def train_gendermodel(self, model_type='vgg16'):

        # initial face model
        self.gender_vgg_model = vggface_vgg16_gender(input_shape=self.input_shape,model_type=model_type)
        self.gender_vgg_model.compile(loss='categorical_crossentropy', \
                                       optimizer=Adam(lr=1e-3), \
                                       metrics=['accuracy'])

        for layer in self.gender_vgg_model.layers:
            print(layer, layer.trainable)

        checkpointer = ModelCheckpoint(filepath='face_gender_model.h5', \
                                       monitor='val_loss', \
                                       verbose=1, \
                                       save_best_only=True, \
                                       mode='auto'
                                       )

        loss_track = TensorBoard(log_dir='./face_gender_estimator_logs', histogram_freq=0, \
                                 batch_size=self.batch_size, \
                                 write_graph=True, \
                                 write_grads=False, \
                                 write_images=False, \
                                 embeddings_freq=0, \
                                 embeddings_layer_names=None, \
                                 embeddings_metadata=None, \
                                 embeddings_data=None, \
                                 update_freq='epoch')
        # for i in range(0, epochs):
        #     print("epoch ", i)
        #     batch_train = np.random.choice(self.train_x.shape[0], size=batch_size)
        #     score = self.gender_vgg_model.fit(self.train_x[batch_train], self.train_y[batch_train], \
        #                                   epochs=1, \
        #                                   validation_data=(self.test_x, self.test_y), \
        #                                   callbacks=[checkpointer])

        score = self.gender_vgg_model.fit(self.train_x, self.train_y, \
                                           batch_size=self.batch_size, \
                                           epochs=self.epochs, \
                                           validation_data=(self.test_x, self.test_y), \
                                           callbacks=[checkpointer, loss_track])

        self.gender_vgg_model = load_model("face_gender_model.h5")  # optional: load the best model before test
        self.gender_vgg_model.save_weights('face_gender_model_weights.h5')
        print("Gender Training is done")

if __name__ == '__main__':

    args = getarg();
    train_model = args.train_model
    print("Ready to training:", train_model)

    input_shape = (224,224,3)
    dataload = data_load()
    df = dataload.dataOperation()
    estimator = estimator(input_shape=input_shape,testsize=0.3,epochs=250, batch_size=80)
    estimator.trainingDataPrepared()
    if (train_model == "age"):
       estimator.train_agemodel()
    elif(train_model=="gender"):
        model_tpye = 'vgg16' #or resnet50 please make sure the accurate spell.
        estimator.train_gendermodel(model_tpye)