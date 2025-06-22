import cv2
from cvzone.HandTrackingModule import HandDetector

# ===== Configuration =====
CAM_WIDTH = 1280       # Reduced resolution for better performance
CAM_HEIGHT = 720
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_HANDS = 2          # Maximum number of hands to detect
# =========================

def count_fingers(lmList, hand_type):
    fingers = []
    
    # Thumb detection (different logic for left/right hand)
    if hand_type == "Right":
        thumb_open = lmList[4][0] > lmList[3][0]  # Right hand thumb
    else:
        thumb_open = lmList[4][0] < lmList[3][0]  # Left hand thumb
    fingers.append(1 if thumb_open else 0)
    
    # Other fingers (index to pinky)
    tip_ids = [8, 12, 16, 20]  # Finger tip landmarks
    for tip in tip_ids:
        fingers.append(1 if lmList[tip][1] < lmList[tip-2][1] else 0)
    
    return sum(fingers)

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_WIDTH)
    cap.set(4, CAM_HEIGHT)
    
    # Initialize detector
    detector = HandDetector(
        maxHands=MAX_HANDS,
        detectionCon=MIN_DETECTION_CONFIDENCE,
        minTrackCon=MIN_TRACKING_CONFIDENCE
    )
    
    while True:
        # Read frame
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        
        # Detect hands
        hands, img = detector.findHands(img, flipType=True)
        
        if hands:
            for hand in hands:
                lmList = hand["lmList"]
                bbox = hand["bbox"]
                hand_type = hand["type"]
                
                # Skip invalid hand data
                if not lmList or len(lmList) < 21:
                    continue
                
                # Count fingers
                finger_count = count_fingers(lmList, hand_type)
                
                # Draw hand-specific information
                color = (0, 255, 0) if hand_type == "Right" else (0, 0, 255)
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20),
                            (bbox[0]+bbox[2]+20, bbox[1]+bbox[3]+20),
                            color, 2)
                
                # Display finger count near hand
                text_pos = (bbox[0]-50, bbox[1]-50 if bbox[1]-50 > 50 else bbox[1]+50)
                cv2.putText(img, f"{hand_type}: {finger_count}", 
                        text_pos, cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
        
        # Add FPS counter
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(img, f"FPS: {int(fps)}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display image
        cv2.imshow("Hand Tracking", img)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()