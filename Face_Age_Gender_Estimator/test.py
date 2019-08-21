import cv2
import os
import numpy as np
import argparse
import shutil
import os

from keras.models import load_model
from pathlib import Path
from Gender_Model import *

def getarg():

    parser = argparse.ArgumentParser(description="get basic parameters for test",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, default='./test',
                        help="test images directory")
    parser.add_argument("--output_dir", type=str, default='./output', help="show the result of gender or age")

    args = parser.parse_args()
    return args

def load_image(img_dir):
    img_dir = Path(img_dir)
    for img_path in img_dir.glob("*.*"):
        img_array = cv2.imread(str(img_path),1)
        if img_array is not None:
            img = cv2.resize(img_array, (224,224))
            img = np.expand_dims(img, axis=0)
            img = img/255
            yield img, img_path

def main():
    #collect data
    args = getarg();
    image_dir = args.image_dir
    output_dir = args.output_dir

    img_arraywithpath = load_image(image_dir)
    # gender
    # load model and weights (gender and age model)
    # gender_vgg_model = genderModel() #mode type = vgg16 by default or resnet50
    gender_vgg_model = load_model("./pretrained/gender_model.h5")
    gender_vgg_model.load_weights('./pretrained/gender_model_weights.h5')
    age_vgg_model = load_model("./pretrained/age_model.h5")
    age_vgg_model.load_weights("./pretrained/age_model_weights.h5")

    for img_array, path in img_arraywithpath:
        if str(path).find('\\'):
            newpath = str(path).replace('\\', '/')
        filename = newpath.split('/')[-1]
        pred_gender = gender_vgg_model.predict(img_array)
        gender = "M" if np.argmax(pred_gender) == 1 else "F"
        pred_age = age_vgg_model.predict(img_array)
        age_indexes = np.array([i for i in range(0, 101)])
        age = int(np.sum(pred_age * age_indexes, axis=1))
        newfilename = "gender_{}_age_{}_{}".format(gender, age, filename) #label on the image file name
        shutil.copy(path, output_dir)
        os.rename(os.path.join(output_dir,filename), os.path.join(output_dir,newfilename))

if __name__ == '__main__':
    main()
    print("Done! please check the output folder")