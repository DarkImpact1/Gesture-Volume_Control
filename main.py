import cv2 as cv
import time
from trackingModule import HandDetector  # Import the HandDetector class from handtrackingModule

def main():
    width_camera, height_camera = 640, 480  # Set camera resolution
    previous_time = 0  # To calculate FPS

    # Initialize webcam
    camera = cv.VideoCapture(0)
    camera.set(3, width_camera)  # Set width of the camera
    camera.set(4, height_camera)  # Set height of the camera

    # Create an instance of HandDetector from the imported module
    handDetector = HandDetector(detectionConfidence=0.8)

    while True:
        # Capture frame from the webcam
        success, cam_image = camera.read()
        if not success:
            print("Failed to capture image")
            break

        # Use the findHand function to detect hands in the current frame
        cam_image = handDetector.findHand(cam_image, draw=True)

        # Get the positions of the landmarks (fingers, hand, etc.)
        landmark_positions = handDetector.findPosition(cam_image, draw=False)

        # Control volume based on the distance between thumb and index finger
        cam_image, volume_bar, volume_percentage = handDetector.controlVolume(cam_image, landmark_positions)

        # Display the volume bar if available
        if volume_bar:
            cv.rectangle(cam_image, (50, 150), (85, 400), (0, 255, 0), 2)
            cv.rectangle(cam_image, (50, int(volume_bar)), (85, 400), (0, 255, 0), cv.FILLED)
            cv.putText(cam_image, f"{int(volume_percentage)}%", (55, 390), 1, 0.9, (0, 0, 0), 2)

        # Calculate and display the FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv.putText(cam_image, f"{int(fps)} fps", (10, 24), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # Show the webcam feed with hand tracking and volume control
        cv.imshow("Hand Gesture Volume Control", cam_image)

        # Break the loop when 'q' is pressed
        if cv.waitKey(1) == ord('q'):
            break

    # Release the camera and close OpenCV windows
    camera.release()
    cv.destroyAllWindows()

# Entry point of the program
if __name__ == "__main__":
    main()
