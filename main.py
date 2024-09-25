import cv2 as cv
import mediapipe as mp
import time

camera =  cv.VideoCapture(0)


# hand detection module it will locate all the points of hands in cameraand it accepts RGB image
mpHands = mp.solutions.hands
hands = mpHands.Hands() # by default it detects only two hands
mpDraw = mp.solutions.drawing_utils

# variable to detect frame rate 
previous_time, current_time = 0,0
while True:
    success, cam_image = camera.read()
    RGB_image = cv.cvtColor(cam_image,cv.COLOR_BGR2RGB)
    mp_hand_result = hands.process(RGB_image)
    hand_landmarks = mp_hand_result.multi_hand_landmarks
    # if camera is detecting hands only then 
    if hand_landmarks:
        # draw landmarks on each fist 
        for fist in hand_landmarks:
            #let's detect the index and landmark from fist
            for index,landmarks in enumerate(fist.landmark):
                # print(index,landmarks)
                height, width, channels = cam_image.shape
                pos_x, pos_y = int(landmarks.x*width), int(landmarks.y*height)
                print(index,pos_x,pos_y)
                if index == 4:
                    cv.circle(cam_image,(pos_x,pos_y),15,(255,255,0),cv.FILLED)


            mpDraw.draw_landmarks(cam_image,fist,mpHands.HAND_CONNECTIONS)
            
    
    # Displaying FPS on screen 
    current_time = time.time()
    fps  = 1/(current_time - previous_time)
    previous_time = current_time
    cv.putText(cam_image,str(int(fps)),(10,70),cv.FONT_HERSHEY_COMPLEX_SMALL,3,(0,0,255),2)

    cv.imshow("Main-camera",cam_image)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()