import mediapipe as mp
import time
import cv2 as cv

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

    def findhand(self,image,draw = False):
        RGB_image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        mp_hand_result = self.hands.process(RGB_image)
        hand_landmarks = mp_hand_result.multi_hand_landmarks
        # if camera is detecting hands only then 
        if hand_landmarks:
            # draw landmarks on each fist 
            for fist in hand_landmarks:
                # #let's detect the index and landmark from fist
                # for index,landmarks in enumerate(fist.landmark):
                #     # print(index,landmarks)
                #     height, width, channels = image.shape
                #     pos_x, pos_y = int(landmarks.x*width), int(landmarks.y*height)
                #     print(index,pos_x,pos_y)
                #     if index == 4:
                #         cv.circle(image,(pos_x,pos_y),15,(255,255,0),cv.FILLED)

                if draw:
                    self.mpDraw.draw_landmarks(image,fist,self.mpHands.HAND_CONNECTIONS)


def main():
    camera =  cv.VideoCapture(0)
    # variable to detect frame rate 
    previous_time, current_time = 0,0
    hand_detector = HandDetector() 
    while True:
        success, cam_image = camera.read()   
        hand_detector.findhand(cam_image,True)

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