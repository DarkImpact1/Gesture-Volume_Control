import mediapipe as mp
import time
import cv2 as cv
import numpy as np
import math
import pycaw
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# HandDetector class is responsible for detecting hand landmarks using MediaPipe and controlling system volume 
# by calculating the distance between the thumb and index finger. The volume control is achieved through the Pycaw library.
class HandDetector():
    def __init__(self, mode=False, maxhands=2, modelComplexity=1, detectionConfidence=0.5, trackingConfidence=0.5):
        """
        Initialize the HandDetector class with parameters for MediaPipe hand tracking and Pycaw for volume control.
        
        :param mode: Whether to track continuously or just process each frame independently.
        :param maxhands: Maximum number of hands to detect.
        :param modelComplexity: Complexity of the hand model (higher complexity gives better accuracy but slower performance).
        :param detectionConfidence: Minimum confidence value for hand detection to be considered successful.
        :param trackingConfidence: Minimum confidence value for hand tracking to be considered successful.
        """
        # MediaPipe hands module and drawing utilities initialization
        self.mode = mode
        self.maxHands = maxhands
        self.detectionConfidence = detectionConfidence
        self.trackinConfidence = trackingConfidence
        self.modelComplexity = modelComplexity

        # Set up the MediaPipe Hands solution
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackinConfidence)
        self.mpDraw = mp.solutions.drawing_utils  # Utility to draw hand landmarks

        # Pycaw library for controlling system volume
        # Get system audio output device and initialize volume control interface
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = self.interface.QueryInterface(IAudioEndpointVolume)
        self.volume.GetMasterVolumeLevel()
        self.volume_range = self.volume.GetVolumeRange()  # Get the min and max volume levels
        self.min_volume = self.volume_range[0]  # Minimum system volume
        self.max_volume = self.volume_range[1]  # Maximum system volume
        self.volume_bar = 400  # Starting height for the volume bar display
        self.volume_percentage = 0  # Current volume percentage

    # Method to detect hands and optionally draw landmarks
    def findHand(self, image, draw=False):
        """
        Detect hands in a given image frame and optionally draw landmarks.

        :param image: Input image frame from the camera.
        :param draw: Boolean flag to indicate whether to draw the hand landmarks on the image.
        :return: Image with or without drawn hand landmarks.
        """
        # Convert BGR image to RGB as MediaPipe works with RGB images
        RGB_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Process the image to detect hand landmarks
        self.mp_hand_result = self.hands.process(RGB_image)
        hand_landmarks = self.mp_hand_result.multi_hand_landmarks

        # If hands are detected and drawing is enabled, draw landmarks on the image
        if hand_landmarks:
            for fist in hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, fist, self.mpHands.HAND_CONNECTIONS)

        return image

    # Method to extract positions of detected hand landmarks
    def findPosition(self, image, handNo=0, draw=True):
        """
        Extract and return positions of detected hand landmarks.

        :param image: Input image frame from the camera.
        :param handNo: Index to select which hand to analyze (in case multiple hands are detected).
        :param draw: Boolean flag to indicate whether to draw landmark indices on the image.
        :return: List of (landmark index, x-position, y-position) for each detected landmark.
        """
        landmarkList = []  # List to store the positions of the hand landmarks
        hand_landmarks = self.mp_hand_result.multi_hand_landmarks

        # If landmarks are found, extract their coordinates
        if hand_landmarks:
            my_fist_1 = hand_landmarks[handNo]

            # Loop through all landmarks on the hand
            for index, landmarks in enumerate(my_fist_1.landmark):
                height, width, channels = image.shape
                # Convert landmark normalized coordinates to image pixel coordinates
                pos_x, pos_y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([index, pos_x, pos_y])

                # Optionally draw the index number of each landmark on the image
                if draw:
                    cv.putText(image, str(index), (pos_x, pos_y), 1, 1, (0, 0, 0), 1)

        return landmarkList

    # Method to control system volume based on hand gesture (distance between thumb and index finger)
    def controlVolume(self, image, landmark_positions):
        """
        Control system volume based on the distance between the thumb and index finger.
        Displays volume bar and adjusts system volume accordingly.

        :param image: Input image frame from the camera.
        :param landmark_positions: List of landmark positions to calculate the distance between thumb and index finger.
        :return: Updated image with volume bar, and the calculated volume percentage and volume bar height.
        """
        if len(landmark_positions) != 0:
            # Extract coordinates of the thumb (landmark 4) and index finger (landmark 8)
            thumb_x, thumb_y = landmark_positions[4][1], landmark_positions[4][2]
            index_x, index_y = landmark_positions[8][1], landmark_positions[8][2]
            # Midpoint between thumb and index finger
            center_x, center_y = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2

            # Draw circles and a line between the thumb and index finger for visualization
            cv.circle(image, (thumb_x, thumb_y), 5, (255, 255, 0), cv.FILLED)
            cv.circle(image, (index_x, index_y), 5, (255, 255, 0), cv.FILLED)
            cv.circle(image, (center_x, center_y), 5, (0, 0, 0), cv.FILLED)
            cv.line(image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 555), 3)

            # Calculate the Euclidean distance between the thumb and index finger
            distance = math.hypot(thumb_x - index_x, thumb_y - index_y)

            # Convert hand distance range to volume control range and update the volume bar and percentage
            self.volume_bar = np.interp(distance, [20, 150], [400, 150])
            self.volume_percentage = np.interp(distance, [20, 150], [0, 100])
            converted_volume = np.interp(self.volume_percentage, [0, 100], [-20, self.max_volume])

            # Further refine the volume control for specific distance ranges
            if distance < 85 and distance > 51:
                converted_volume = np.interp(distance, [0, 85], [-35, -11])
            elif distance < 52 and distance > 20:
                converted_volume = np.interp(distance, [0, 52], [-65, -22])
            elif distance < 20:
                converted_volume = self.min_volume  # Set to minimum volume when fingers are too close

            # Set the system volume based on the calculated volume level
            self.volume.SetMasterVolumeLevel(converted_volume, None)

        return [image, self.volume_bar, self.volume_percentage]


# Main function to capture webcam feed, detect hand gestures, and control volume in real-time
def main():
    width_camera, height_camera = 640, 480  # Set the resolution for the camera feed

    # Variables to calculate FPS (Frames Per Second)
    previous_time = 0

    # Open the webcam and set the resolution
    camera = cv.VideoCapture(0)
    camera.set(3, width_camera)  # Set the width
    camera.set(4, height_camera)  # Set the height

    # Create an instance of the HandDetector class
    handDetector = HandDetector(detectionConfidence=0.8)

    # Main loop to process the camera feed
    while True:
        success, cam_image = camera.read()  # Capture frame from the webcam

        # Detect hands and their landmarks
        cam_image = handDetector.findHand(cam_image, True)
        landmark_positions = handDetector.findPosition(cam_image, draw=False)

        # Control volume based on hand gestures (thumb and index finger positions)
        cam_image, volume_bar, volume_percentage = handDetector.controlVolume(cam_image, landmark_positions)

        # Display the volume bar on the image
        if volume_bar:
            cv.rectangle(cam_image, (50, 150), (85, 400), (0, 255, 0), 2)
            cv.rectangle(cam_image, (50, int(volume_bar)), (85, 400), (0, 255, 0), cv.FILLED)
            cv.putText(cam_image, f"{int(volume_percentage)}%", (55, 390), 1, 0.9, (0, 0, 0), 2)

        # Calculate FPS and display it on the screen
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv.putText(cam_image, f"{int(fps)} fps", (10, 24), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # Show the processed image in a window
        cv.imshow("Main-camera", cam_image)

        # Break the loop and exit when 'q' is pressed
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cv.destroyAllWindows()


# Entry point of the program
if __name__ == "__main__":
    main()
