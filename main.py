import cv2 as cv
import mediapipe as mp
import time

camera =  cv.VideoCapture(0)
while True:
    success, image = camera.read()
    cv.imshow("Main-camera",image)
    cv.waitKey(1)