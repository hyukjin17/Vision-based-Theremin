# Vision-Based Theremin

A touchless musical instrument controlled entirely by hands in front of a webcam (no sensors, no MIDI controller, no physical hardware beyond a camera). The right hand controls pitch (and selects chord qualities via finger pinches), while the left hand controls volume and audio effects through learned gesture recognition.

---

## Demo

![Demo Video Link](https://youtu.be/krMZuCWujUQ)

---

## How It Works

The system is a four-stage real-time pipeline that runs once per webcam frame at ~30 FPS:

1. **Capture** — OpenCV pulls a frame from the webcam and mirror-flips it
2. **Detect** — MediaPipe Hands extracts 21 3D landmarks per hand and labels each as "Left" or "Right"
3. **Classify** — Right-hand pinches are detected by thresholded fingertip distances; left-hand poses are fed through a small MLP that picks one of 6 gesture classes
4. **Synthesize** — Pyo builds a real-time DSP graph (oscillators → distortion → bitcrush → delay) with smoothed parameter updates

End-to-end latency is not perceptible in practice. The dominant contribution is the camera frame interval (~33 ms); MediaPipe inference and MLP inference are each well under a millisecond on CPU.

---

## Features

- **No hardware beyond a webcam** — works on any laptop with a built-in or USB camera
- **Vintage Tone** - blends a clean Sine wave (70%) with a Triangle LFO (30%) for a warm, vintage synthesizer sound
- **Smooth audio** — all parameters interpolated through `SigTo` ramps + exponential smoothing on pitch, so there are no clicks when gestures change (bridges the gap between 30 FPS webcam tracking and 44.1kHz audio processing)
- **Parallel Signal Routing** - Glitch/Bitcrush effect utilizes an `Interp` crossfader running in parallel to the main signal chain (guarantees a pure, uncolored tone when the effect is bypassed)
- **Custom data collection tool** for retraining the classifier on your own gestures
- **~99% validation accuracy** on the six-class classifier, trained in under a minute on CPU
- **Confidence gating** on the classifier — if it's not sure, it holds the previous mode rather than firing a spurious mode switch

### Right Hand: Pitch & Chord Selection
* **Logarithmic Pitch Control:** The vertical (Y-axis) position of the wrist controls the fundamental pitch. Logarithmic scaling ensures that equal physical movements result in equal musical intervals across a 2-octave range (200Hz - 800Hz).
* **Pinch-to-Chord Heuristics:** Euclidean distance calculations between fingertips detect complex pinches, instantly fanning the monophonic oscillator out into 3-note chords.
  * **Major:** Thumb + Index
  * **Minor:** Thumb + Middle
  * **Sus4:** Thumb + Ring
  * **Power (Root-5th-8ve):** Thumb + Pinky
  * **Diminished:** Thumb + Index + Middle
  * **Augmented:** Thumb + Middle + Ring
  * **Major 7th:** Thumb + Ring + Pinky

### Left Hand: Effects Controller
A custom PyTorch Multi-Layer Perceptron (MLP) classifies the left hand into specific gestures, acting as an intelligent state machine for audio effects.
* **OPEN PALM:** Controls Master Volume (Y-axis).
* **FIST:** Controls Waveform Distortion/Drive (Y-axis).
* **POINT (INDEX):** Controls a 2D Echo/Delay.
* **PEACE SIGN:** Controls a destructive 8-bit Glitch/Bitcrush effect (Y-axis controls sample rate degradation).
* **THREE FINGERS:** Holding this gesture for 1 second toggles between **Continuous Mode** (always playing) and **Chord-Only Mode** (only playing chords when pinch motion is detected). It also resets all effect parameters.

The classifier requires **≥85% confidence** to switch modes, so transitions between gestures won't accidentally trigger mode changes.

---

## Requirements

- Python 3.10 or newer
- A webcam (built-in or USB)
- Audio output (speakers or headphones)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hyukjin17/Vision-based-Theremin.git
cd Vision-based-Theremin
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on `pyo`:** On macOS you may need to install PortAudio first via `brew install portaudio`. On Linux, install `python3-dev`, `libasound2-dev`, and `portaudio19-dev` through your package manager. Windows users should install the Pyo pre-built wheel from [the Pyo downloads page](http://ajaxsoundstudio.com/software/pyo/) if `pip install pyo` fails.

### 4. Verify the model file is present

The trained classifier weights should be at `gesture_model.pth` in the project root. If not, see [Training Your Own Gesture Classifier](#training-your-own-gesture-classifier) below.

---

## Quick Start

```bash
python theremin.py
```

This launches the theremin. A window titled "Visual Theremin" will open showing your webcam feed. Position both hands in frame, and you'll hear a tone. Move the right hand up and down to control pitch, and use the left hand to control volume and effects. Real-time note/chord readouts and an effects dashboard should appear on the top left corner. For example:

```
Note: A4 Major (440Hz)
Mode: DISTORTION
Vol:   72%
Dist:  58%
Delay:  0%
Glitch: 0%
```

Press **`q`** to quit.

---

## Training Your Own Gesture Classifier

If you want to use your own gestures, you can retrain in three steps.

### Step 1 — Collect data

```bash
python gesture_data_collector.py
```

A window opens with your webcam feed. Hold a gesture in front of the camera, then press a **digit key (0–5)** to record **300 samples** of that class. The counter on screen tracks progress; when it hits 300, the burst ends automatically.

| Key | Class | Gesture |
| --- | --- | --- |
| `0` | Unknown | transitional/random poses |
| `1` | Volume | open hand |
| `2` | Distortion | closed fist |
| `3` | Delay | pointing index |
| `4` | Glitch | peace sign |
| `5` | Reset | three fingers |

Pro tips:
- Vary your hand's position, finger position, and slight rotations **within each burst** — this makes the classifier robust to natural performance variation
- For class 0 (Unknown), record a mix of in-between and nonsense gestures so the classifier learns to reject them
- If you mess up a burst, edit `gesture_dataset.csv` manually and remove the bad rows

Press `q` to exit when done. Your data is appended to `gesture_dataset.csv`.

### Step 2 — Split into train/test

```bash
python split_data.py
```

This produces `gesture_dataset_split.csv` with a stratified 90/10 train/test split, fixed random seed for reproducibility.

### Step 3 — Train the model

```bash
python train.py
```

Training runs for 100 epochs (usually under a minute on CPU). The final weights are saved to `gesture_model.pth` and a training curve plot to `loss_plot.png`.

The theremin will automatically pick up the new weights the next time you run `theremin.py`.

---

## Project Structure

```
vision-theremin/
├── theremin.py                  # Main application — run this
├── gesture_data_collector.py    # Record training samples
├── split_data.py                # Train/test split (90/10 stratified)
├── train.py                     # Train the MLP classifier
├── model.py                     # Network architecture (63→64→32→6 MLP)
├── dataset.py                   # PyTorch Dataset wrapper
│
├── gesture_model.pth            # Trained classifier weights
├── gesture_dataset.csv          # Raw collected landmark data
├── gesture_dataset_split.csv    # Split CSV with train/test labels
├── loss_plot.png                # Training curves (generated)
│
├── requirements.txt
├── README.md                    # This file
└── LICENSE
```

---

## Technical Details

### Hand tracking — MediaPipe

Each hand gives us **21 landmarks x 3 coordinates = 63 features**.

### Landmark normalization

1. **Translation-invariant**: subtract landmark 0 (wrist) from every landmark
2. **Scale-invariant**: divide all coordinates by the maximum absolute value

### Training configuration

- Loss: cross-entropy
- Optimizer: AdamW, learning rate 1e-3, default weight decay
- Batch size: 64
- Epochs: 100 (no early stopping; validation loss is already flat by epoch 20)
- Train/test split: 90/10, stratified by class, seed = 42

### Audio synthesis — Pyo

Three frequency-interpolated sine oscillators feed the chord voices (with identical frequencies when playing a single note). The LFO-shaped triangle wave adds a bit of harmonic richness to the otherwise pure sine tone.

**Why the `SigTo` ramps matter**: raw per-frame parameter updates at 30 FPS produce audible clicks (zipper noise). Every control parameter: pitch, volume, distortion drive, delay feedback, bitcrush ratio, glitch is wrapped in a `SigTo` interpolator with an 80 ms ramp. The pitch gets an additional first-order exponential smoothing (α = 0.15) on top of that. Together, these hide the frame-rate jitter and make the system sound continuous.