import mediapipe as mp
import cv2 as cv
import time
import math
import trackingModule as tm
import numpy as np
import pycaw
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# global variable for fps
previous_time = 0



def displayFPS(image):
    global previous_time
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv.putText(image,f"{int(fps)} fps",(10,24),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)

def main():
    ###############################################
    width_camera, height_camera = 640,480
    ###########################################

    camera = cv.VideoCapture(0)
    camera.set(3,width_camera)
    camera.set(4,height_camera)


    # object to detect fist
    handDetector = tm.HandDetector(detectionConfidence=0.8)



    while True:
        success, cam_image = camera.read()
        cam_image = handDetector.findHand(cam_image,True)
        landmark_positions = handDetector.findPosition(cam_image,draw=False)
        cam_image,volume_bar,volume_percentage = handDetector.controlVolume(cam_image,landmark_positions)

        if volume_bar:
            # Display volume change through rectange
            cv.rectangle(cam_image,(50,150),(85,400),(0,255,0),2)
            cv.rectangle(cam_image,(50,int(volume_bar)),(85,400),(0,255,0),cv.FILLED)
            cv.putText(cam_image,f"{int(volume_percentage)}%",(55,390),1,0.9,(0,0,0),2)

        displayFPS(cam_image)
        cv.imshow("Main-camera",cam_image)
        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()