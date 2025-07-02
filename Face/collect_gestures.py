import cv2
import mediapipe as mp
import pandas as pd
import keyboard
import os

def collect_gesture_data():
    """
    Collect hand landmark data for different gestures.
    Press SPACE to capture current gesture, 'q' to quit and save.
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    
    # Initialize data storage
    all_data = []
    all_labels = []
    
    # Define gestures to collect
    gestures = ["thumbs_up", "peace", "open_palm", "fist", "ok_sign"]
    
    for gesture in gestures:
        print(f"\n=== Collecting data for gesture: {gesture} ===")
        print("Position your hand and press SPACE to capture samples.")
        print("Collect at least 50 samples per gesture. Press 'q' to move to next gesture.\n")
        
        cap = cv2.VideoCapture(0)
        gesture_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            # Display gesture info on frame
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {gesture_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | Q: Next gesture", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    # Save if spacebar pressed
                    if keyboard.is_pressed("space") and len(landmarks) == 63:
                        all_data.append(landmarks)
                        all_labels.append(gesture)
                        gesture_count += 1
                        print(f"Captured {gesture} sample #{gesture_count}")
            
            cv2.imshow("Gesture Collection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        cap.release()
        print(f"Collected {gesture_count} samples for {gesture}")
    
    cv2.destroyAllWindows()
    
    # Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df["label"] = all_labels
        df.to_csv("gesture_data.csv", index=False)
        print(f"\n✅ Saved {len(all_data)} total samples to gesture_data.csv")
    else:
        print("❌ No data collected!")

if __name__ == "__main__":
    collect_gesture_data()
