import cv2 as cv
import time
from trackingModule import HandDetector

def main():
    """
    Main function to capture webcam feed, detect hand gestures, and control system volume using hand gestures.
    It leverages the HandDetector class to detect hand landmarks and adjust system volume based on the distance
    between the thumb and index finger.
    """
    width_camera, height_camera = 640, 480  # Set the resolution for the camera feed
    previous_time = 0  # Variable to keep track of time for calculating FPS

    # Initialize the webcam feed
    camera = cv.VideoCapture(0)
    camera.set(3, width_camera) 
    camera.set(4, height_camera) 

    # Create an instance of the HandDetector class from the tracking module
    handDetector = HandDetector()

    # Main loop to process each frame of the webcam feed
    while True:
        # Capture the current frame from the webcam
        success, cam_image = camera.read()
        if not success:
            # If frame capture fails, print an error message and exit the loop
            print("Failed to capture image")
            break

        # Use the findHand method from HandDetector to detect hands in the current frame
        # If a hand is detected, it draws landmarks on the hand
        cam_image = handDetector.findHand(cam_image, draw=True)

        # Retrieve the positions of the hand landmarks (fingers, hand, etc.)
        landmark_positions = handDetector.findPosition(cam_image, draw=False)

        # Control the system volume based on the distance between thumb and index finger
        cam_image, volume_bar, volume_percentage = handDetector.controlVolume(cam_image, landmark_positions)

        # If the volume bar is available (i.e., hand is detected), display it on the screen
        if volume_bar:
            cv.rectangle(cam_image, (50, 150), (85, 400), (0, 255, 0), 2)  # Outer rectangle
            cv.rectangle(cam_image, (50, int(volume_bar)), (85, 400), (0, 255, 0), cv.FILLED)  # Filled volume bar
            # Display the current volume percentage as text on the screen
            cv.putText(cam_image, f"{int(volume_percentage)}%", (55, 390), 1, 0.9, (0, 0, 0), 2)

        # Calculate the FPS (Frames Per Second) by determining the time difference between current and previous frame
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        # Display the calculated FPS on the webcam feed
        cv.putText(cam_image, f"{int(fps)} fps", (10, 24), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv.imshow("Hand Gesture Volume Control", cam_image)

        # Check if the 'q' key is pressed; if so, break the loop and exit
        if cv.waitKey(1) == ord('q'):
            break

    camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
