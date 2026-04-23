"""
Hyuk Jin Chung
4/22/2026

Vision-based Theremin Project
Control sound effects and pitch using hand positions and gestures
Right hand controls the pitch and chords and left hand controls the sound effects and volume
Classifier is trained on custom gesture data to identify left hand gestures to switch between different effects
"""

import cv2
import mediapipe as mp
import numpy as np
from pyo import Server, Sine, SawTable, Mix, TableRead, Delay, SigTo
import warnings
import os

# Suppress the Protobuf warning
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# List of notes for lookup
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def get_dist(p1, p2):
    """Calculate Euclidean distance between two 3D mediapipe landmarks"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    # Start audio engine
    s = Server().boot()
    s.start()

    # Audio interpolators (array of 3 SigTo objects to support 3-note chords smoothly)
    freq_ctrls = [SigTo(value=440.0, time=0.08) for _ in range(3)]
    vol_ctrl = SigTo(value=0.0, time=0.08)

    # Create a theremin-style tone: sine + sawtooth mix
    # Oscillators (plugged into the array of freq_ctrls)
    osc1 = Sine(freq=freq_ctrls, mul=0.25)
    saw_waveform = SawTable(order=12).normalize()
    osc2 = TableRead(table=saw_waveform, freq=freq_ctrls, mul=0.1)
    combined = Mix([osc1, osc2], voices=1)

    # Effects
    # ----------------------------------------------------------------
    echo = Delay(combined, delay=0.25, feedback=0, mul=0.5).out()
    # ----------------------------------------------------------------
    
    combined.out()

    # Mediapipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
    cap = cv2.VideoCapture(0)

    # Smoothing and mapping variables
    current_base_pitch = 440.0
    current_vol = 0.0
    current_fback = 0.0
    alpha = 0.15 # smoothing percentage
    pinch_threshold = 0.05 # how close the fingers have to be to register as contact

    print("Theremin active")
    print("Right Hand: Pitch | Left Hand: Volume")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # Mirror image to counteract mirroring on webcam
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Default to silent if no hands
            target_vol, target_fback = 0.0, 0.0
            chord_ratios = [1.0, 1.0, 1.0] # default: single note
            chord_name = "Single Note"

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw skeleton landmarks on screen
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    label = results.multi_handedness[idx].classification[0].label
                    # Get wrist position (landmark 0)
                    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                    # Squish the active range (do not use bottom 5% or top 10% of screen)
                    # Invert so higher hand = higher value
                    pos = np.clip(1.0 - ((wrist_y - 0.40) / 0.55), 0.0, 1.0)

                    # Right hand controls (pitch and chords)
                    if label == "Right":
                        # Logarithmic pitch mapping (2x increase in frequency is 1 linear octave increase)
                        # Maps 100Hz - 1600Hz (4 octaves)
                        target_base_pitch = 100.0 * (2 ** (pos * 4))

                        # Chord sounds
                        # Pinch Detection (between thumb and other fingers)
                        thumb = hand_landmarks.landmark[4]
                        d_index = get_dist(thumb, hand_landmarks.landmark[8])
                        d_middle = get_dist(thumb, hand_landmarks.landmark[12])
                        d_ring = get_dist(thumb, hand_landmarks.landmark[16])
                        d_pinky = get_dist(thumb, hand_landmarks.landmark[20])

                        if d_index < pinch_threshold:
                            # Major Triad (Root, Major 3rd, Perfect 5th)
                            chord_ratios, chord_name = [1.0, 1.25, 1.5], "Major"
                        elif d_middle < pinch_threshold:
                            # Minor Triad (Root, Minor 3rd, Perfect 5th)
                            chord_ratios, chord_name = [1.0, 1.189, 1.5], "Minor"
                        elif d_ring < pinch_threshold:
                            # Suspended 4th (Root, Perfect 4th, Perfect 5th)
                            chord_ratios, chord_name = [1.0, 1.333, 1.5], "Sus4"
                        elif d_pinky < pinch_threshold:
                            # Power Chord (Root, Perfect 5th, Octave)
                            chord_ratios, chord_name = [1.0, 1.5, 2.0], "Power"

                        # Apply smoothing to the base pitch
                        current_base_pitch = (alpha * target_base_pitch) + ((1.0 - alpha) * current_base_pitch)
                        
                        # Update Audio Engine (send the 3 frequencies to the SigTo controllers)
                        for i in range(3):
                            freq = float(current_base_pitch * chord_ratios[i])
                            freq_ctrls[i].setValue(freq)

                    # Left hand controls (volume and effects)
                    if label == "Left":
                        # --------------------------------------------------
                        # features = normalize_landmarks(hand_landmarks)
                        # prediction = gesture_classifier(features)
                        # current_mode = CLASS_NAMES[prediction]
                        # --------------------------------------------------

                        # Hardcoded default until ML is trained
                        current_mode = "OPEN_PALM"

                        # The Audio Router
                        if current_mode == "OPEN_PALM":
                            target_vol = pos * 0.3
                        elif current_mode == "FIST":
                            # target_distortion = pos
                            pass
                        elif current_mode == "POINT":
                            # target_fback = pos * 0.9
                            pass
                        elif current_mode == "PEACE": # vowel filter / phaser
                            pass
                        elif current_mode == "RESET": # reset to default
                            pass

            # Update Volume Audio Engine
            current_vol = (alpha * target_vol) + ((1.0 - alpha) * current_vol)
            vol_ctrl.setValue(float(current_vol * 2.0))
            
            # ----------------------------------------------------------------------
            # Optional: Hook up effects (dormant right now)
            current_fback = (alpha * target_fback) + ((1.0 - alpha) * current_fback)
            echo.setFeedback(float(current_fback))
            echo.setMul(vol_ctrl)
            combined.setMul(vol_ctrl)
            # ----------------------------------------------------------------------

            # Display current state (UI)
            safe_pitch = max(current_base_pitch, 1.0)
            midi_note = int(np.round(12 * np.log2(safe_pitch / 440.0) + 69))
            note_name = NOTE_NAMES[midi_note % 12]
            octave = (midi_note // 12) - 1

            if chord_name == "Single Note":
                display_note = f"{note_name}{octave}"
            else:
                display_note = f"{note_name}{octave} {chord_name}"

            cv2.putText(frame, f"Note: {display_note} ({int(current_base_pitch)}Hz)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Vol:  {int(current_vol*333)}%", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Show active left-hand mode
            cv2.putText(frame, f"Mode: OPEN_PALM", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow('Visual Theremin', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        s.stop()

if __name__ == "__main__":
    main()