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
import torch
import time
from pyo import Server, Sine, LFO, Mix, Delay, SigTo, Disto, Interp, Degrade
import warnings
import os

from model import HandGestureNet
from gesture_data_collector import normalize_landmarks

# Suppress the Protobuf warning
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# List of notes for lookup
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Gesture Dictionary
CLASS_NAMES = {
    0: "UNKNOWN",
    1: "VOLUME",
    2: "DISTORTION",
    3: "DELAY/ECHO",
    4: "GLITCH",
    5: "RESET"
}

def get_dist(p1, p2):
    """Calculate Euclidean distance between two 3D mediapipe landmarks"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    """
    Run the vision-based theremin program
    - assign pitch and chords to right hand and volume and effects to the left hand
    - use the classifier to identify left hand gestures (open hand, closed hand, peace sign, etc...)
    - use simple distance threshold to detect right hand pinch gestures
    """

    # Load gesture classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HandGestureNet().to(device)
    try:
        model.load_state_dict(torch.load('gesture_model.pth', map_location=device))
        model.eval()
        print("Successfully loaded Gesture Classifier")
        print(f"Running inference on: {device}")
    except FileNotFoundError:
        print("ERROR: 'gesture_model.pth' not found. Train the model first")
        return

    # Start audio engine
    s = Server().boot().start()

    # Audio interpolators (array of 3 SigTo objects to support 3-note chords smoothly)
    freq_ctrls = [SigTo(value=440.0, time=0.08) for _ in range(3)]
    vol_ctrl = SigTo(value=0.0, time=0.08)
    # Effects controllers
    dist_ctrl = SigTo(value=0.0, time=0.08)
    delay_ctrl = SigTo(value=0.0, time=0.08)
    # Glitch controls
    bitcrush_ctrl = SigTo(value=1.0, time=0.08)
    glitch_mix_ctrl = SigTo(value=0.0, time=0.08)

    # Create a theremin-style tone (sine + triangle)
    # Oscillators (plugged into the array of freq_ctrls)
    osc1 = Sine(freq=freq_ctrls, mul=0.25)
    osc2 = LFO(freq=freq_ctrls, type=3, mul=0.1)
    combined = Mix([osc1, osc2], voices=1)

    # Audio Effects
    # ----------------------------------------------------------------
    # Effect 1: Distortion
    dist = Disto(combined, drive=dist_ctrl, slope=0.8)
    
    # Effect 2: Glitch effect (runs in parallel to avoid 'polluting' the sound)
    bitcrush = Degrade(dist, bitdepth=6, srscale=bitcrush_ctrl)

    # Crossfader (blend between the clean and destroyed (bitcrush) signals)
    master_out = Interp(dist, bitcrush, glitch_mix_ctrl, mul=vol_ctrl).out()

    # Effect 3: 
    echo = Delay(master_out, delay=0.5, feedback=delay_ctrl).out()
    # ----------------------------------------------------------------

    # State variables
    target_vol = 0.0
    current_vol = 0.0
    target_dist = 0.0
    target_delay = 0.0
    target_bitcrush = 1.0
    target_glitch_mix = 0.0

    current_mode = "UNKNOWN"
    current_base_pitch = 440.0
    alpha = 0.15 # pitch smoothing percentage
    pinch_threshold = 0.10 # how close the fingers have to be to register as contact

    # Toggle Mode Variables
    continuous_mode = True
    last_toggle_time = 0.0

    # Mediapipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
    cap = cv2.VideoCapture(0)

    print("Theremin active")
    print("Right Hand: Pitch | Left Hand: Volume / Effects")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # Mirror image to counteract mirroring on webcam
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            chord_ratios = [1.0, 1.0, 1.0] # default: single note
            chord_name = "Single Note"

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw skeleton landmarks on screen
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    label = results.multi_handedness[idx].classification[0].label

                    # Get wrist position (landmark 0)
                    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                    # Squish the active range (do not use bottom 5% or top 40% of screen)
                    # Invert so higher hand = higher value
                    pos = np.clip(1.0 - ((wrist_y - 0.40) / 0.55), 0.0, 1.0)

                    # ======================================================================
                    # Right hand controls (pitch and chords)
                    if label == "Right":
                        # Logarithmic pitch mapping (2x increase in frequency is 1 linear octave increase)
                        # Maps 2 octaves
                        target_base_pitch = 200.0 * (2 ** (pos * 2))

                        # Chord sounds
                        # Pinch Detection (between thumb and other fingers)
                        thumb = hand_landmarks.landmark[4]
                        d_index = get_dist(thumb, hand_landmarks.landmark[8])
                        d_middle = get_dist(thumb, hand_landmarks.landmark[12])
                        d_ring = get_dist(thumb, hand_landmarks.landmark[16])
                        d_pinky = get_dist(thumb, hand_landmarks.landmark[20])

                        # Check for Thumb + Index + Middle (Diminished)
                        if d_index < pinch_threshold and d_middle < pinch_threshold:
                            # Diminished triad (Root, Minor 3rd, Diminished 5th)
                            chord_ratios, chord_name = [1.0, 1.189, 1.414], "Diminished"
                        # Check for Thumb + Middle + Ring (Augmented)
                        elif d_middle < pinch_threshold and d_ring < pinch_threshold:
                            # Augmented triad (Root, Major 3rd, Augmented 5th)
                            chord_ratios, chord_name = [1.0, 1.25, 1.587], "Augmented"
                        # Check for Thumb + Ring + Pinky (Major 7th)
                        elif d_ring < pinch_threshold and d_pinky < pinch_threshold:
                            # Major 7th (Root, Major 3rd, Major 7th)
                            chord_ratios, chord_name = [1.0, 1.25, 1.888], "Major 7th"
                        elif d_index < pinch_threshold:
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

                    # ======================================================================
                    # Left hand controls (volume and effects)
                    if label == "Left":
                        features = normalize_landmarks(hand_landmarks)
                        tensor_features = torch.tensor([features], dtype=torch.float32).to(device)

                        with torch.no_grad():
                            outputs = model(tensor_features)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            confidence, pred_class = torch.max(probs, 1)
                            
                        class_idx = pred_class.item()

                        # Classifier Confidence Threshold
                        if confidence.item() < 0.85 or class_idx == 0:
                            current_mode = "UNKNOWN"
                        else:
                            current_mode = CLASS_NAMES[class_idx]

                        # The Audio Router
                        if current_mode == "VOLUME":
                            target_vol = pos * 0.4 # volume control
                            target_glitch_mix = 0.0 # mute the glitch channel
                        elif current_mode == "DISTORTION":
                            target_dist = pos * 0.95 # max 95% distortion
                            target_glitch_mix = 0.0
                        elif current_mode == "DELAY/ECHO":
                            target_delay = pos * 0.85 # max 85% feedback
                            target_glitch_mix = 0.0
                        elif current_mode == "GLITCH":
                            target_bitcrush = 1.0 - (pos * 0.95)
                            target_glitch_mix = 1.0 # unmute glitch
                        elif current_mode == "RESET": # reset to default
                            # Check if 1 second has passed since the last toggle
                            current_time = time.time()
                            if current_time - last_toggle_time > 1.0:
                                continuous_mode = not continuous_mode # Toggle the mode
                                last_toggle_time = current_time
                            # Reset all effects
                            target_dist = 0.0
                            target_delay = 0.0
                            target_bitcrush = 1.0
                            target_glitch_mix = 0.0

            # Audio gating (based on mode)
            # If in continuous mode OR if a chord is actively being pinched, use the target volume
            # Otherwise, override the volume to 0.0 to mute the instrument
            if continuous_mode or chord_name != "Single Note":
                active_vol = target_vol
            else:
                active_vol = 0.0

            # Apply smoothing so the sound glides on and off like a real instrument
            current_vol = (alpha * active_vol) + ((1.0 - alpha) * current_vol)

            # Update Audio Controllers with makeup gain
            vol_ctrl.setValue(float(current_vol * 2.0))
            dist_ctrl.setValue(float(target_dist))
            bitcrush_ctrl.setValue(float(target_bitcrush))
            delay_ctrl.setValue(float(target_delay))
            glitch_mix_ctrl.setValue(float(target_glitch_mix))

            # Display current state (UI)
            safe_pitch = max(current_base_pitch, 1.0)
            midi_note = int(np.round(12 * np.log2(safe_pitch / 440.0) + 69))
            note_name = NOTE_NAMES[midi_note % 12]
            octave = (midi_note // 12) - 1

            if chord_name == "Single Note":
                display_note = f"{note_name}{octave}"
            else:
                display_note = f"{note_name}{octave} {chord_name}"

            cv2.putText(frame, f"Note: {display_note} ({int(current_base_pitch)}Hz)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Mode: {current_mode}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # Effects Dashboard
            cv2.putText(frame, f"Vol: {int(target_vol*250)}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Dist: {int(target_dist*105)}%", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Delay: {int(target_delay*117)}%", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            glitch_percent = int((1.0 - target_bitcrush) * 105)
            cv2.putText(frame, f"Glitch: {glitch_percent}%", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Display current playing style
            style_text = "CONTINUOUS" if continuous_mode else "CHORD ONLY"
            cv2.putText(frame, f"Play Style: {style_text}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Visual Theremin', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'): break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        s.stop()

if __name__ == "__main__":
    main()