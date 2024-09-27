import mediapipe as mp
import time
import cv2 as cv
import numpy as np
import math
import pycaw
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# HandDetector class to detect hands and control system volume based on finger distance
class HandDetector():
    def __init__(self, mode=False, maxhands=2, modelComplexity=1, detectionConfidence=0.5, trackingConfidence=0.5):
        # Initialize parameters for MediaPipe hand detection
        self.mode = mode
        self.maxHands = maxhands
        self.detectionConfidence = detectionConfidence
        self.trackinConfidence = trackingConfidence
        self.modelComplexity = modelComplexity

        # Initialize MediaPipe Hands module and Drawing Utilities
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackinConfidence)
        self.mpDraw = mp.solutions.drawing_utils

        # Initialize Pycaw for controlling system volume
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = self.interface.QueryInterface(IAudioEndpointVolume)
        self.volume.GetMasterVolumeLevel()
        self.volume_range = self.volume.GetVolumeRange()  # Get min and max volume levels
        self.min_volume = self.volume_range[0]
        self.max_volume = self.volume_range[1]
        self.volume_bar = 400  # Initial height of the volume bar
        self.volume_percentage = 0

    # Method to detect hand landmarks in the image
    def findHand(self, image, draw=False):
        RGB_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert BGR image to RGB
        self.mp_hand_result = self.hands.process(RGB_image)  # Process the image to detect hands
        hand_landmarks = self.mp_hand_result.multi_hand_landmarks

        # If hand landmarks are detected, optionally draw them
        if hand_landmarks:
            for fist in hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, fist, self.mpHands.HAND_CONNECTIONS)

        return image

    # Method to find the positions of landmarks on the hand
    def findPosition(self, image, handNo=0, draw=True):
        landmarkList = []  # List to store positions of landmarks
        hand_landmarks = self.mp_hand_result.multi_hand_landmarks

        # If hand landmarks are detected, extract their positions
        if hand_landmarks:
            my_fist_1 = hand_landmarks[0]

            # Loop through each landmark on the hand and append its coordinates
            for index, landmarks in enumerate(my_fist_1.landmark):
                height, width, channels = image.shape
                pos_x, pos_y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([index, pos_x, pos_y])

                # Optionally draw the index of each landmark on the image
                if draw:
                    cv.putText(image, str(index), (pos_x, pos_y), 1, 1, (0, 0, 0), 1)

        return landmarkList

    # Method to control system volume based on the distance between thumb and index finger
    def controlVolume(self, image, landmark_positions):
        if len(landmark_positions) != 0:
            # Get coordinates of thumb (landmark 4) and index finger (landmark 8)
            thumb_x, thumb_y = landmark_positions[4][1], landmark_positions[4][2]
            index_x, index_y = landmark_positions[8][1], landmark_positions[8][2]
            center_x, center_y = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2  # Midpoint between thumb and index

            # Draw circles on thumb, index finger, and midpoint, and a line between thumb and index finger
            cv.circle(image, (thumb_x, thumb_y), 5, (255, 255, 0), cv.FILLED)
            cv.circle(image, (index_x, index_y), 5, (255, 255, 0), cv.FILLED)
            cv.circle(image, (center_x, center_y), 5, (0, 0, 0), cv.FILLED)
            cv.line(image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 555), 3)

            # Calculate distance between thumb and index finger
            distance = math.hypot(thumb_x - index_x, thumb_y - index_y)

            # Convert hand range (20-150 pixels) to volume bar height (400-150) and volume percentage (0-100%)
            self.volume_bar = np.interp(distance, [20, 150], [400, 150])
            self.volume_percentage = np.interp(distance, [20, 150], [0, 100])
            converted_volume = np.interp(self.volume_percentage, [0, 100], [-20, self.max_volume])

            # Adjust the volume more finely based on distance
            if distance < 85 and distance > 51:
                converted_volume = np.interp(distance, [0, 85], [-35, -11])
            elif distance < 52 and distance > 20:
                converted_volume = np.interp(distance, [0, 52], [-65, -22])
            elif distance < 20:
                converted_volume = self.min_volume  # Minimum volume for very close fingers

            # Set the system volume based on calculated level
            self.volume.SetMasterVolumeLevel(converted_volume, None)

        return [image, self.volume_bar, self.volume_percentage]


# Main function to run the hand detection and volume control in real-time
def main():
    width_camera, height_camera = 640, 480  # Set camera resolution

    # Variables to track FPS (Frames Per Second)
    previous_time = 0

    # Open webcam for capturing video
    camera = cv.VideoCapture(0)
    camera.set(3, width_camera)
    camera.set(4, height_camera)

    # Create an instance of the HandDetector class
    handDetector = HandDetector(detectionConfidence=0.8)

    # Main loop to process the webcam feed
    while True:
        success, cam_image = camera.read()  # Capture frame from the webcam

        # Find hands and hand landmarks in the current frame
        cam_image = handDetector.findHand(cam_image, True)
        landmark_positions = handDetector.findPosition(cam_image, draw=False)

        # Control volume based on finger positions
        cam_image, volume_bar, volume_percentage = handDetector.controlVolume(cam_image, landmark_positions)

        # Display the volume bar on the screen
        if volume_bar:
            cv.rectangle(cam_image, (50, 150), (85, 400), (0, 255, 0), 2)
            cv.rectangle(cam_image, (50, int(volume_bar)), (85, 400), (0, 255, 0), cv.FILLED)
            cv.putText(cam_image, f"{int(volume_percentage)}%", (55, 390), 1, 0.9, (0, 0, 0), 2)

        # Calculate and display the FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv.putText(cam_image, f"{int(fps)} fps", (10, 24), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # Show the webcam feed
        cv.imshow("Main-camera", cam_image)

        # Break the loop when 'q' is pressed
        if cv.waitKey(1) == ord('q'):
            break

    # Release camera and close OpenCV windows
    cv.destroyAllWindows()


# Entry point of the program
if __name__ == "__main__":
    main()
