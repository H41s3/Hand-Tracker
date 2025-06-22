import cv2
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from collections import deque

# ===== Configuration =====
CAM_WIDTH = 1280
CAM_HEIGHT = 720
GESTURE_COLORS = {
    'I Love You 🤟': (255, 0, 255),
    'Mood 😈': (255, 69, 0),
    'Metal Horns 🤘': (255, 215, 0),
    'Phone 🤙': (0, 255, 255),
    'Spidey 🕷️': (255, 0, 0),
    'Thumbs Up 👍': (0, 255, 0),
    'Thumbs Down 👎': (0, 0, 255),
    'Gun 🔫': (100, 100, 100),
    'Unknown': (255, 255, 255)
}
# =========================

def count_fingers(lmList, hand_type):
    fingers = []
    if len(lmList) < 21:
        return [False]*5
    
    # Thumb check
    thumb_tip = (lmList[4][0], lmList[4][1])
    thumb_base = (lmList[3][0], lmList[3][1])
    thumb_up = thumb_tip[0] > thumb_base[0] if hand_type == "Right" else thumb_tip[0] < thumb_base[0]
    fingers.append(thumb_up)
    
    # Other fingers
    for tip in [8, 12, 16, 20]:
        if tip-2 >= len(lmList):
            fingers.append(False)
            continue
        fingers.append(lmList[tip][1] < lmList[tip-2][1])
    
    return fingers

def detect_gesture(finger_states, lmList):
    # I Love You 🤟 (Thumb + Index + Pinky up)
    if (finger_states[0] and   # Thumb
        finger_states[1] and   # Index
        not finger_states[2] and  # Middle
        not finger_states[3] and  # Ring
        finger_states[4]):     # Pinky
        return 'I Love You 🤟'
    
    # Metal Horns 🤘 (Index + Pinky up)
    if finger_states[1] and finger_states[4] and not any(finger_states[2:4]):
        return 'Metal Horns 🤘'
    
    # Phone 🤙 (Thumb + Pinky up)
    if finger_states[0] and finger_states[4] and not any(finger_states[1:4]):
        return 'Phone 🤙'
    
    # Spidey Web 🕷️ (Middle + Ring down)
    if finger_states[1] and not finger_states[2] and not finger_states[3] and finger_states[4]:
        return 'Spidey 🕷️'
    
    # Gun 🔫 (Index up + thumb up)
    if finger_states[0] and finger_states[1] and not any(finger_states[2:]):
        return 'Gun 🔫'
    
    # Thumbs Up/Down 👍👎
    if sum(finger_states[1:]) == 0:
        return 'Thumbs Up 👍' if finger_states[0] else 'Thumbs Down 👎'
    
    # Mood 😈 (Middle finger only)
    if finger_states[2] and not any([finger_states[1], finger_states[3], finger_states[4]]):
        return 'Mood 😈'
    
    return 'Unknown'

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)
    
    detector = HandDetector(maxHands=2, detectionCon=0.9)
    gesture_history = deque(maxlen=15)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            time.sleep(0.1)
            continue
            
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, draw=True)
        effect_img = img.copy()

        try:
            if hands:
                for hand in hands:
                    if not all(key in hand for key in ["lmList", "bbox", "type"]):
                        continue
                        
                    lmList = hand["lmList"]
                    bbox = hand["bbox"]
                    hand_type = hand["type"]
                    
                    if len(lmList) < 21:
                        continue
                    
                    # Convert to 2D points
                    lmList_2d = [[point[0], point[1]] for point in lmList]
                    
                    finger_states = count_fingers(lmList, hand_type)
                    gesture = detect_gesture(finger_states, lmList)
                    gesture_history.append(gesture)
                    
                    # I Love You Effect
                    if gesture == 'I Love You 🤟':
                        # Draw heart between thumb and pinky
                        pt1 = (int(lmList_2d[4][0]), int(lmList_2d[4][1]))
                        pt2 = (int(lmList_2d[20][0]), int(lmList_2d[20][1]))
                        cv2.line(effect_img, pt1, pt2, (255,0,255), 3)
                        cv2.putText(effect_img, "❤️", (bbox[0]-50, bbox[1]-100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 3)
                    
                    # ... (keep other effects the same) ...

            # Blend effects
            alpha = 0.7
            img = cv2.addWeighted(img, 1-alpha, effect_img, alpha, 0)
            
            # Get stable gesture
            stable_gesture = max(set(gesture_history), 
                            key=lambda x: list(gesture_history).count(x)) if gesture_history else 'Unknown'
            
            cv2.putText(img, stable_gesture, 
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, 
                    GESTURE_COLORS.get(stable_gesture, (255,255,255)), 5)
            
            cv2.imshow("Gesture Control", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()