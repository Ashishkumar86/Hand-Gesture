import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
#from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import DepthwiseConv2D


#model_path = r"D:\Sign Language\converted_keras (5)\keras_model.h5"
#labels_path = r"D:\Sign Language\converted_keras (5)\labels.txt"

#model = load_model(model_path, custom_objects=custom_objects)


# Load the model, ensuring custom objects are handled if necessary
#custom_objects = {'DepthwiseConv2D': DepthwiseConv2D}
#model = load_model('path/to/your/model.h5', custom_objects=custom_objects)


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(r"D:\Sign Language\converted_keras (5)\keras_model.h5", r"D:\Sign Language\converted_keras (5)\labels.txt")

#classifier = Classifier("D:\Sign Language\converted_keras (5)\keras_model.h5" , "D:\Sign Language\converted_keras (5)\labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["Hello","I Love You","Ok","Thank You"]  # labels should be in order

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h 
            wCal = math.ceil(k * w)
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            print(prediction, index)

        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize, hCal))
            imgResizeShape  = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap : hCal + hGap,:] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter +=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg' , imgWhite) # type: ignore
        print(counter)
