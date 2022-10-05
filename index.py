### Importing Libraries ###
# linear algebra
import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

#Visiulazation
import matplotlib.pyplot as plt

#image processing
import cv2

class FaceDetector():

    def __init__(self, faceCascadePath):
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

    def detect(self, image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        # function return rectangle coordinates of faces for given image
        rects = self.faceCascade.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        return rects

#Frontal face of haar cascade loaded
frontal_cascade_path="haarcascade_frontalface_default.xml"

#Detector object created
fd = FaceDetector(frontal_cascade_path)

#An image contains faces, loaded
#image_org = cv2.imread("1.png")
## another Image
image_org = cv2.imread("2.jpg")


def get_faces():
    return np.copy(image_org)

def show_image(image):
    plt.figure(figsize=(18, 15))
    #Before showing image, bgr color order transformed to rgb order
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
show_image(get_faces())


def detect_face(image, scaleFactor, minNeighbors, minSize):
    # face will detected in gray image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = fd.detect(image_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

    for x, y, w, h in faces:
        # detected faces shown in color image
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 255, 0), 3)

    show_image(image)


image_output = get_faces()
detect_face(image=image_output, scaleFactor=1.3, minNeighbors=3, minSize=(30,30))










