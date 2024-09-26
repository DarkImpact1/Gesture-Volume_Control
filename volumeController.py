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
min_volume = volume_range[0]
max_volume = volume_range[1]
volume_bar = 400 # Height of rectange
volume_percentage = 0

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
        # Hand range is from 10 to  and we have to convert it into -65 to 0
        print(distance)
        converted_volume = np.interp(distance,[10,120],[min_volume,max_volume])
        volume_bar = np.interp(distance,[10,120],[400,150]) # minimum should be at 400 and max should be at 150 i.e height
        volume_percentage = np.interp(distance,[10,120],[0,100])
        volume.SetMasterVolumeLevel(converted_volume, None)

        if int(distance) < 15:  
            cv.circle(cam_image,(center_x,center_y),5,(255,255,255),cv.FILLED)
 
        # print(distance)


        # print(landmark_positions[4],landmark_positions[8])

    # Display volume change through rectange
    cv.rectangle(cam_image,(50,150),(85,400),(0,255,0),2)
    cv.rectangle(cam_image,(50,int(volume_bar)),(85,400),(0,255,0),cv.FILLED)
    cv.putText(cam_image,str(int(volume_percentage)),(55,390),1,0.9,(0,0,0),2)


    displayFPS(cam_image)
    cv.imshow("Main-camera",cam_image)
    if cv.waitKey(1) == ord('q'):
        break


cv.destroyAllWindows()