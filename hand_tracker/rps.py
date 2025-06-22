import cv2
import time
from cvzone.HandTrackingModule import HandDetector
from collections import deque

# ===== Configuration =====
CAM_WIDTH = 1280
CAM_HEIGHT = 720
HISTORY_LENGTH = 10  # Number of frames for gesture stabilization
GESTURE_COLORS = {
    'Rock ✊': (0, 0, 255),
    'Paper 🖐️': (0, 255, 0),
    'Scissors ✌️': (255, 0, 0),
    'Unknown': (255, 255, 255)
}
# =========================

def count_fingers(lmList, hand_type):
    fingers = []
    
    # Thumb detection
    if hand_type == "Right":
        thumb_open = lmList[4][0] > lmList[3][0]
    else:
        thumb_open = lmList[4][0] < lmList[3][0]
    fingers.append(thumb_open)
    
    # Other fingers
    tip_ids = [8, 12, 16, 20]
    for tip in tip_ids:
        fingers.append(lmList[tip][1] < lmList[tip-2][1])
    
    return sum(fingers)

def detect_gesture(finger_count, lmList):
    # Rock-Paper-Scissors Logic
    if finger_count == 0:
        return 'Rock ✊'
    elif finger_count == 5:
        return 'Paper 🖐️'
    elif finger_count == 2:
        # Verify scissors position (index and middle fingers up)
        if lmList[8][1] < lmList[6][1] and lmList[12][1] < lmList[10][1]:
            return 'Scissors ✌️'
    return 'Unknown'

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)
    
    detector = HandDetector(maxHands=2, detectionCon=0.8, minTrackCon=0.5)
    gesture_history = deque(maxlen=HISTORY_LENGTH)
    pTime = 0  # For FPS calculation
    trail_points = []  # For movement trail effect

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        
        # Flip image horizontally for mirror effect
        img = cv2.flip(img, 1)
        
        hands = detector.findHands(img, draw=True, flipType=False)
        current_gestures = []

        if hands:
            for hand in hands:
                lmList = hand["lmList"]
                bbox = hand["bbox"]
                hand_type = hand["type"]
                
                if not lmList or len(lmList) < 21:
                    continue
                
                # Count fingers and detect gesture
                finger_count = count_fingers(lmList, hand_type)
                gesture = detect_gesture(finger_count, lmList)
                current_gestures.append(gesture)
                
                # Get stable gesture from history
                gesture_history.append(gesture)
                stable_gesture = max(set(gesture_history), 
                                   key=lambda x: list(gesture_history).count(x))
                
                # Draw hand-specific elements
                color = GESTURE_COLORS.get(stable_gesture, (255,255,255))
                
                # Bounding box
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20),
                            (bbox[0]+bbox[2]+20, bbox[1]+bbox[3]+20),
                            color, 3)
                
                # Gesture text
                cv2.putText(img, stable_gesture, 
                          (bbox[0]-50, bbox[1]-50 if bbox[1]-50 > 50 else bbox[1]+50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                
                # Movement trail effect
                wrist_pos = (lmList[0][0], lmList[0][1])
                trail_points.append(wrist_pos)
                if len(trail_points) > 20:
                    trail_points.pop(0)
                
                # Draw trail
                for i, point in enumerate(trail_points):
                    cv2.circle(img, point, 5-i//4, color, cv2.FILLED)

        # Gesture history panel
        cv2.rectangle(img, (10, 10), (300, 50 + 30*HISTORY_LENGTH), (40,40,40), -1)
        for i, gesture in enumerate(gesture_history):
            y = 40 + i*30
            cv2.putText(img, f"{i+1}. {gesture}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (CAM_WIDTH-200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow("Gesture Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()