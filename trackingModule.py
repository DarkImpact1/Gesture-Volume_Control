import mediapipe as mp
import time
import cv2 as cv
import numpy as np
import math
import pycaw
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



class HandDetector():
    def __init__(self,mode=False, maxhands=2,modelComplexity = 1, detectionConfidence = 0.5, trackingConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxhands
        self.detectionConfidence = detectionConfidence
        self.trackinConfidence = trackingConfidence
        self.modelComplexity = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplexity,self.detectionConfidence,self.trackinConfidence)
        self.mpDraw = mp.solutions.drawing_utils

        # Audio variables
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = self.interface.QueryInterface(IAudioEndpointVolume)
        self.volume.GetMasterVolumeLevel()
        self.volume_range = self.volume.GetVolumeRange()
        self.min_volume = self.volume_range[0]
        self.max_volume = self.volume_range[1]
        self.volume_bar = 400 # Height of rectange
        self.volume_percentage = 0

    def findHand(self,image,draw = False):
        RGB_image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        self.mp_hand_result = self.hands.process(RGB_image)
        hand_landmarks = self.mp_hand_result.multi_hand_landmarks
        # if camera is detecting hands only then 
        if hand_landmarks:
            # draw landmarks on each fist 
            for fist in hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image,fist,self.mpHands.HAND_CONNECTIONS)

        return image
    

    def findPosition(self,image,handNo=0,draw = True):
        landmarkList = []
        hand_landmarks = self.mp_hand_result.multi_hand_landmarks
        # if camera is detecting hands only then 
        if hand_landmarks:
            my_fist_1 = hand_landmarks[0]

            #let's detect the index and landmark from fist
            for index,landmarks in enumerate(my_fist_1.landmark):
                # print(index,landmarks)
                height, width, channels = image.shape
                pos_x, pos_y = int(landmarks.x*width), int(landmarks.y*height)
                # print(index,pos_x,pos_y)
                landmarkList.append([index,pos_x,pos_y])
                if draw:
                    cv.putText(image,str(index),(pos_x,pos_y),1,1,(0,0,0),1)
                        # cv.circle(image,(pos_x,pos_y),15,(255,255,0),cv.FILLED)

        return landmarkList

    def controlVolume(self,image,landmark_positions):
        
        if len(landmark_positions) != 0:
            thumb_x,thumb_y = landmark_positions[4][1],landmark_positions[4][2]
            index_x,index_y = landmark_positions[8][1],landmark_positions[8][2]
            center_x, center_y = (thumb_x + index_x)//2, (thumb_y + index_y)//2

            cv.circle(image,(thumb_x,thumb_y),5,(255, 255, 0),cv.FILLED)
            cv.circle(image,(index_x,index_y),5,( 255, 255, 0),cv.FILLED)
            cv.circle(image,(center_x,center_y),5,(0,0,0),cv.FILLED)
            cv.line(image,(thumb_x,thumb_y),(index_x, index_y),(255,0,555),3)     

            distance = math.hypot(thumb_x-index_x,thumb_y-index_y)
            # Hand range is from 10 to  and we have to convert it into -65 to 0
            # print(distance)
            converted_volume = np.interp(distance,[10,120],[self.min_volume,self.max_volume])
            self.volume_bar = np.interp(distance,[10,120],[400,150]) # minimum should be at 400 and max should be at 150 i.e height
            self.volume_percentage = np.interp(distance,[10,120],[0,100])
            self.volume.SetMasterVolumeLevel(converted_volume, None)

        return [image,self.volume_bar,self.volume_percentage]
# This function is for the testing purpose

def main():
    camera =  cv.VideoCapture(0)
    # variable to detect frame rate 
    previous_time, current_time = 0,0
    hand_detector = HandDetector() 
    while True:
        success, cam_image = camera.read()   
        detected_fist = hand_detector.findHand(cam_image,True)
        landmark_positions = hand_detector.findPosition(detected_fist)
        print(landmark_positions)

        # Displaying FPS on screen 
        current_time = time.time()
        fps  = 1/(current_time - previous_time)
        previous_time = current_time
        cv.putText(cam_image,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX_SMALL,3,(0,0,255),2)

        # command to exit the camera 
        cv.imshow("Main-camera",cam_image)
        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()