import cv2 as cv
import mediapipe as mp
import time
import  tracking_module as tm



detector = tm.HandDetector()
camera =  cv.VideoCapture(0)
# variable to detect frame rate 
previous_time, current_time = 0,0
while True:
    success, cam_image = camera.read()
    detected_hands = detector.findHand(image=cam_image,draw=True)
    landmark_positions = detector.findPosition(image=cam_image)    
    # print(landmark_positions)
    
    # Displaying FPS on screen 
    current_time = time.time()
    fps  = 1/(current_time - previous_time)
    previous_time = current_time
    cv.putText(cam_image,str(int(fps))+"fps",(10,70),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)

    cv.imshow("Main-camera",cam_image)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()