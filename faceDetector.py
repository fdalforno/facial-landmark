import cv2
import urllib.request as urlreq
import os

haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
haarcascade = "haarcascade_frontalface_alt2.xml"

haarcascade_eyes_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
haarcascade_eyes = "haarcascade_eye.xml"

LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"
LBFmodel = "lbfmodel.yaml"

left_eye = 41
rigth_eye = 46

nose_upper = 27
nose_lower = 8

left_mouth = 60
right_mounth = 64

interesing_landmark = {
    'eyes_orizontal' : (left_eye,rigth_eye),
    'nose_vertical' : (nose_upper,nose_lower),
    'mouth_orizontal' : (left_mouth,right_mounth)
}



def detectFace(image_gray):
    if (haarcascade in os.listdir(os.curdir)):
        print("File exists")
    else:
        #scarico il modello del detect
        urlreq.urlretrieve(haarcascade_url, haarcascade)
        print("File downloaded")

    #creo il classificatore
    detector = cv2.CascadeClassifier(haarcascade)

    #estraggo lo facce dal classificatore
    faces = detector.detectMultiScale(image_gray)

    #mostro le coordinate
    print("Faces:\n", faces)

   
    return faces


def detectEyes(image_gray):
    if (haarcascade_eyes in os.listdir(os.curdir)):
        print("File exists")
    else:
        #scarico il modello del detect
        urlreq.urlretrieve(haarcascade_eyes_url, haarcascade_eyes)
        print("File downloaded")

    #creo il classificatore
    detector = cv2.CascadeClassifier(haarcascade_eyes)

    #estraggo lo facce dal classificatore
    eyes = detector.detectMultiScale(image_gray)

    #mostro le coordinate
    print("eyes:\n", eyes)

   
    return eyes


def getLandmark(image_gray,faces):
    if (LBFmodel in os.listdir(os.curdir)):
        print("File exists")
    else:
        #scarico il modello del detect
        urlreq.urlretrieve(LBFmodel_url, LBFmodel)
        print("File downloaded")
    
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(LBFmodel)
    _, landmarks = landmark_detector.fit(image_gray, faces)
    return landmarks

def extractLandmark(landmarks):
    landmarks_dictionary = {}
    for landmark in landmarks:
        points = landmark[0]
        for key in interesing_landmark:
            start,end = interesing_landmark[key]
            start_point = points[start]
            end_point = points[end]
            landmarks_dictionary[key] = (start_point,end_point)
    return landmarks_dictionary
            

