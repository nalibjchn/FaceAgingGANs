from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("--root_dir", required=True,
                help="path to root directories of input images")
ap.add_argument("--des_dir", required=True,
                help="path to destination directories of output images")
ap.add_argument("--notdetect_dir", required=True,
                help="path to not detect face image directoriew of input images")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=400)

root_dir = args["root_dir"]
des_dir = args["des_dir"].split("/")[-1]
notdetect_dir = args["notdetect_dir"].split("/")[-1]

if not os.path.exists(notdetect_dir):
    os.mkdir('../'+ notdetect_dir)

input_files = [os.path.join(dp,f) for dp, dn, fn in os.walk(os.path.expanduser(root_dir)) for f in fn]

# loop over the face detections
for input_file in input_files:
    try:
        print(input_file)
        image = cv2.imread(input_file)
        image = imutils.resize(image, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        des_path = '/'.join(['..',des_dir] + input_file.split("/")[2:-1])
        des_list = des_path.split("/")
        for i in range(len(des_list)):
            elem = des_list[:i+1]
            if elem and not os.path.exists("/".join(elem)):
                os.mkdir("/".join(elem))
        file_name = input_file.split("/")[-1]
        out_file = os.path.join(des_path,file_name)
        # show the original input image and detect faces in the grayscale

        rects = detector(gray, 2)
        if len(rects) < 1:
            image = cv2.imread(input_file)
            if os.path.isfile(input_file):
                file_name = input_file.split("/")[-1]
                print("not detect:" + file_name)
                notdetect_file = os.path.join('../'+notdetect_dir, file_name)
                cv2.imwrite(notdetect_file,image)
        else:
            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                try:
                    (x, y, w, h) = rect_to_bb(rect)
                    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=400) #sync all dataset 400 * 400 pixel
                    faceAligned = fa.align(image, gray, rect)

                    # display the output images
                    if os.path.isfile(out_file):
                        file_name = out_file.split("/")[-1]
                        print(file_name.split(".")[0]+file_name.split(".")[1])
                        out_file = os.path.join(des_path, file_name.split(".")[0]+"."+file_name.split(".")[-1])
                    cv2.imwrite(out_file,faceAligned)
                    cv2.waitKey(0)
                    print(out_file)
                except:
                     # display the output images
                    print("CANNOT SAVE_begin")
                    notdetect_file = os.path.join('../'+notdetect_dir, file_name)
                    cv2.imwrite(notdetect_dir,faceAligned)
                    print("CANNOT SAVE_end")
                    continue
    except:
        pass
