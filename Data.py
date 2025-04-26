import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)  # Use 0 for web camera, 1 for external camera
detector = HandDetector(maxHands=1)  # Only show one hand at a time
offset = 20
imgSize = 300
counter = 0

folder = "D:\Sign Language\Data\Ok"
while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # x-axis, y-axis, width, height

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        # Check if the cropped image is valid
        if imgCrop.size > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                # Working with height
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal+wGap] = imgResize  # This is a dictionary

            else:
                # Working with width
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap+hCal, :] = imgResize  # This is a dictionary

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

      #  else:
           # print("Invalid Crop Region")

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)  # Collect data from the keyboard
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
