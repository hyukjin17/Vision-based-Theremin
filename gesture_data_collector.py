"""
Hyuk Jin Chung
4/22/2026

Data collection script to gather hand landmarks data for the MLP gesture classifier
Collects 300 samples continuously for each button press (button press is class-specific and is used for labeling data)
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os

def normalize_landmarks(landmarks):
    """Centers at wrist and normalizes scale (scale invariant)"""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    wrist = coords[0]
    relative_coords = coords - wrist
    
    max_value = np.max(np.abs(relative_coords))
    if max_value > 0:
        normalized_coords = relative_coords / max_value
    else:
        normalized_coords = relative_coords
        
    # Flatten from (21, 3) to a 1D array of 63 elements
    return normalized_coords.flatten().tolist()

def main():
    """Run data collection"""
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
    cap = cv2.VideoCapture(0)

    csv_file = "gesture_dataset.csv"
    file_exists = os.path.isfile(csv_file)
    
    # Target samples per burst
    TARGET_SAMPLES = 300
    
    # State variables
    recording_class = None
    samples_collected = 0
    count_dict = {str(i): 0 for i in range(6)}

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['label'] + [f'lm_{i}' for i in range(63)]
            writer.writerow(header)

        print("Data Collection Ready")
        print(f"Press keys 0-5 to start a {TARGET_SAMPLES}-sample burst")
        print("Press 'q' to quit")

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            key = cv2.waitKey(1) & 0xFF
            char_key = chr(key) if key < 256 else ""

            # Check if user triggered a new burst
            if char_key in "012345" and recording_class is None:
                recording_class = int(char_key)
                samples_collected = 0
                print(f"\nStarted collecting {TARGET_SAMPLES} samples for Class {recording_class}")

            left_hand_visible = False

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label
                    
                    if label == "Left":
                        left_hand_visible = True
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        if recording_class is not None:
                            features = normalize_landmarks(hand_landmarks)
                            writer.writerow([recording_class] + features)
                            
                            samples_collected += 1
                            count_dict[str(recording_class)] += 1
                            
                            # On-screen progress UI
                            cv2.putText(frame, f"Recording Class {recording_class}: {samples_collected}/{TARGET_SAMPLES}", 
                                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            
                            # Check if the burst is complete
                            if samples_collected >= TARGET_SAMPLES:
                                print(f"Success: Collected exactly {TARGET_SAMPLES} valid samples for Class {recording_class}!")
                                recording_class = None # Reset state machine

            # Handle the case where we are recording but the hand drops off screen
            if recording_class is not None and not left_hand_visible:
                cv2.putText(frame, "Waiting for Left Hand...", (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            # Display Overall Stats
            stats = " | ".join([f"{k}:{v}" for k, v in count_dict.items()])
            cv2.putText(frame, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Data Collector', frame)
            if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()