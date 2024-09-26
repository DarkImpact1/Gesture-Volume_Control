import mediapipe as mp
import cv2 as cv
import time
import math
import trackingModule as tm
import numpy as np
import pycaw
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def displayFPS(image):
    global previous_time
    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time
    cv.putText(cam_image,f"{int(fps)} fps",(10,24),cv.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)


###############################################
width_camera, height_camera = 640,480
###########################################

camera = cv.VideoCapture(0)
camera.set(3,width_camera)
camera.set(4,height_camera)


# global variable for fps
previous_time = 0

# Audio 

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None)
min_volume = volume_range[0]
max_volume = volume_range[1]

# object to detect fist
handDetector = tm.HandDetector(detectionConfidence=0.8)

while True:
    success, cam_image = camera.read()

    cam_image = handDetector.findHand(cam_image,True)
    landmark_positions = handDetector.findPosition(cam_image,draw=False)
    if len(landmark_positions) != 0:
        thumb_x,thumb_y = landmark_positions[4][1],landmark_positions[4][2]
        index_x,index_y = landmark_positions[8][1],landmark_positions[8][2]
        center_x, center_y = (thumb_x + index_x)//2, (thumb_y + index_y)//2

        cv.circle(cam_image,(thumb_x,thumb_y),5,(255, 255, 0),cv.FILLED)
        cv.circle(cam_image,(index_x,index_y),5,( 255, 255, 0),cv.FILLED)
        cv.circle(cam_image,(center_x,center_y),5,(0,0,0),cv.FILLED)
        cv.line(cam_image,(thumb_x,thumb_y),(index_x, index_y),(255,0,555),3)     

        distance = math.hypot(thumb_x-index_x,thumb_y-index_y)
        # Hand range is from 10 to 120
        converted_volume = np.interp(distance,[10,120],[min_volume,max_volume])
        print(converted_volume)

        if int(distance) < 15:  
            cv.circle(cam_image,(center_x,center_y),5,(255,255,255),cv.FILLED)
 
        # print(distance)


        # print(landmark_positions[4],landmark_positions[8])
    displayFPS(cam_image)
    cv.imshow("Main-camera",cam_image)
    if cv.waitKey(1) == ord('q'):
        break


cv.destroyAllWindows()