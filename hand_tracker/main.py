import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1980)  # Set width of the frame
cap.set(4, 1080)  # Set height of the frame

# Create HandDetector object
detector = HandDetector(detectionCon=0.8)

while True:
    # Read frame from the webcam
    success, img = cap.read()

    # Find hands in the frame
    hands, img = detector.findHands(img)

    # Check if hands are found
    if hands:
        # Iterate through detected hands
        for hand in hands:
            # Get hand landmarks
            lmList = hand["lmList"]

            # Initialize finger count
            finger_count = 0

            # Check thumb
            if lmList[4][0] > lmList[3][0]:  # Thumb: Compare tip (id 4) with base (id 3)
                finger_count += 1

            # Check other fingers
            for finger_id in range(1, 5):  # Finger IDs from 1 to 4 (index finger to pinky finger)
                # Use the tip of the finger (id 4 * finger_id) and the base of the finger (id 4 * finger_id - 2)
                if lmList[finger_id * 4][1] < lmList[finger_id * 4 - 2][1]:
                    finger_count += 1

            # Print the finger count for each hand
            print("Fingers:", finger_count)

            # Optionally, you can draw text on the image with the finger count
            cv2.putText(img, f"Fingers: {finger_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Hand Tracking", img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
