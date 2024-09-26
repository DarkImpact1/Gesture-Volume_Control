import mediapipe as mp
import time
import cv2 as cv
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class HandDetector():
    """
    A class for detecting hands using MediaPipe and controlling system volume using gesture detection.
    
    Attributes:
    - mode (bool): Static image mode or video stream mode.
    - maxHands (int): Maximum number of hands to detect.
    - detectionConfidence (float): Minimum detection confidence.
    - trackingConfidence (float): Minimum tracking confidence.
    - mpHands: MediaPipe hands object.
    - hands: Initialized hands detection object.
    - mpDraw: MediaPipe drawing utility to draw hand landmarks.
    - volume: Audio control object for managing system volume.
    - min_volume, max_volume (float): Range of system volume.
    - volume_bar (int): Height of the volume bar.
    - volume_percentage (float): Current system volume as a percentage.
    """

    def __init__(self, mode=False, maxhands=2, modelComplexity=1, detectionConfidence=0.5, trackingConfidence=0.5):
        """
        Initializes the HandDetector object with MediaPipe for hand detection and Pycaw for system volume control.
        """
        # MediaPipe hands configuration
        self.mode = mode
        self.maxHands = maxhands
        self.detectionConfidence = detectionConfidence
        self.trackinConfidence = trackingConfidence
        self.modelComplexity = modelComplexity

        # Initialize MediaPipe hands detection
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity,
                                        self.detectionConfidence, self.trackinConfidence)
        self.mpDraw = mp.solutions.drawing_utils

        # Audio variables using Pycaw for controlling system volume
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.volume = self.interface.QueryInterface(IAudioEndpointVolume)
        self.volume.GetMasterVolumeLevel()
        self.volume_range = self.volume.GetVolumeRange()
        self.min_volume = self.volume_range[0]  # Minimum volume level
        self.max_volume = self.volume_range[1]  # Maximum volume level
        self.volume_bar = 400  # Initial height of the volume bar
        self.volume_percentage = 0  # Initial volume percentage

    def findHand(self, image, draw=False):
        """
        Detects hands in a given image.
        
        Args:
        - image: Input image frame (BGR format).
        - draw (bool): Flag to draw hand landmarks on the image.

        Returns:
        - image: The input image with hand landmarks drawn if 'draw' is True.
        """
        # Process the image with MediaPipe hand detection (no need to convert BGR to RGB)
        self.mp_hand_result = self.hands.process(image)
        hand_landmarks = self.mp_hand_result.multi_hand_landmarks

        # If hand landmarks are detected, optionally draw them
        if hand_landmarks:
            for fist in hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, fist, self.mpHands.HAND_CONNECTIONS)
        return image

    def findPosition(self, image, handNo=0, draw=True):
        """
        Extracts the hand landmark positions from the detected hands.
        
        Args:
        - image: Input image frame (BGR format).
        - handNo (int): The index of the hand to process (if multiple hands are detected).
        - draw (bool): Flag to draw landmark indices on the image.

        Returns:
        - landmarkList (list): A list of (index, x, y) positions of the hand landmarks.
        """
        landmarkList = []
        if self.mp_hand_result.multi_hand_landmarks:
            my_fist_1 = self.mp_hand_result.multi_hand_landmarks[handNo]

            # Extract landmark positions in pixels
            for index, landmarks in enumerate(my_fist_1.landmark):
                height, width, channels = image.shape
                pos_x, pos_y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([index, pos_x, pos_y])
                if draw:
                    # Draw the index of each landmark on the image
                    cv.putText(image, str(index), (pos_x, pos_y), 1, 1, (0, 0, 0), 1)
        return landmarkList

    def controlVolume(self, image, landmark_positions):
        """
        Controls system volume based on the distance between the thumb and index finger.
        
        Args:
        - image: Input image frame (BGR format).
        - landmark_positions: A list of hand landmark positions.
        
        Returns:
        - [image, self.volume_bar, self.volume_percentage]: Updated image with volume control visual feedback.
        """
        if len(landmark_positions) != 0:
            # Get positions of thumb (landmark 4) and index finger (landmark 8)
            thumb_x, thumb_y = landmark_positions[4][1], landmark_positions[4][2]
            index_x, index_y = landmark_positions[8][1], landmark_positions[8][2]
            center_x, center_y = (thumb_x + index_x) // 2, (thumb_y + index_y) // 2

            # Visualize thumb and index positions with circles and a line
            cv.circle(image, (thumb_x, thumb_y), 5, (255, 255, 0), cv.FILLED)
            cv.circle(image, (index_x, index_y), 5, (255, 255, 0), cv.FILLED)
            cv.circle(image, (center_x, center_y), 5, (0, 0, 0), cv.FILLED)
            cv.line(image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 555), 3)

            # Calculate the distance between the thumb and index finger
            distance = math.hypot(thumb_x - index_x, thumb_y - index_y)

            # Convert hand range to system volume range
            converted_volume = np.interp(distance, [10, 120], [self.min_volume, self.max_volume])
            self.volume_bar = np.interp(distance, [10, 120], [400, 150])  # Update volume bar height
            self.volume_percentage = np.interp(distance, [10, 120], [0, 100])  # Update volume percentage
            self.volume.SetMasterVolumeLevel(converted_volume, None)

        return [image, self.volume_bar, self.volume_percentage]


# Main function to test hand detection and volume control
def main():
    """
    Captures video from the webcam and applies hand detection and volume control.
    Press 'q' to exit the camera feed.
    """
    camera = cv.VideoCapture(0)
    previous_time, current_time = 0, 0  # For calculating FPS
    frame_skip = 2  # Process every 2nd frame to reduce CPU load
    frame_counter = 0
    hand_detector = HandDetector()  # Initialize the HandDetector object

    while True:
        success, cam_image = camera.read()  # Capture frame from the webcam
        frame_counter += 1

        # Skip every second frame to reduce load
        if frame_counter % frame_skip == 0:
            # Detect hands and get landmark positions
            detected_fist = hand_detector.findHand(cam_image, True)
            landmark_positions = hand_detector.findPosition(detected_fist)

            # Control volume based on hand gesture (if landmarks are found)
            if landmark_positions:
                hand_detector.controlVolume(detected_fist, landmark_positions)

            # Calculate and display FPS every 10 frames
            if frame_counter % 10 == 0:
                current_time = time.time()
                fps = 1 / (current_time - previous_time)
                previous_time = current_time
                cv.putText(cam_image, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 255), 2)

        # Show the camera feed with the hand and volume visualization
        cv.imshow("Main-camera", cam_image)

        # Press 'q' to exit the camera feed
        if cv.waitKey(1) == ord('q'):
            break

    camera.release()  # Release the camera resource
    cv.destroyAllWindows()  # Close all OpenCV windows


if __name__ == "__main__":
    main()
