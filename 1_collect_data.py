"""
STEP 1 - ISL Data Collection
==============================
Run this script to collect training data for each ISL gesture.
It uses MediaPipe to extract landmarks from up to 2 hands (42 landmarks × 3 axes = 126 features).

⚠️  IMPORTANT: This version now supports 2-hand signs!
    - If you previously collected data with 1-hand mode, please DELETE 'data/isl_data.csv'
    - Re-run this script to collect new data with 2-hand support
    - Then run 2_train_model.py to retrain the model

Usage:
    python 1_collect_data.py

Controls:
    Press the label key (e.g., 'a', 'b', ...) to start collecting for that gesture.
    Press 'q' to quit.

NOTE: Requires mediapipe==0.10.14
    pip install mediapipe==0.10.14
"""

import cv2
import mediapipe as mp
import csv
import os
import sys

# ── Version check ──────────────────────────────────────────────────────────────
ver = tuple(int(x) for x in mp.__version__.split('.')[:3])
if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'hands'):
    print(f"\n[ERROR] mediapipe {mp.__version__} does not support the solutions API.")
    print("[FIX]  Run:  pip install mediapipe==0.10.14")
    print("       Then re-run this script.\n")
    sys.exit(1)

# ── ISL Gesture Labels ─────────────────────────────────────────────────────────
GESTURES = {
    'a': 'Hello',
    'b': 'Thank You',
    'c': 'Yes',
    'd': 'No',
    'e': 'Please',
    'f': 'Help',
    'g': 'I Love You',
    'h': 'Good',
}

SAMPLES_PER_GESTURE = 200
DATA_FILE = 'data/isl_data.csv'

# ── MediaPipe Setup ────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def init_csv():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            # 126 features: 21 landmarks × 3 axes × 2 hands
            header = [f'{axis}{i}' for i in range(42) for axis in ['x','y','z']] + ['label']
            writer.writerow(header)
        print(f"[INFO] Created {DATA_FILE} with 2-hand support (126 features)")

def save_landmarks(hand_landmarks_list, label):
    """Save features from up to 2 hands. Pad with zeros if only 1 hand detected."""
    row = []
    # Always create space for 2 hands (42 landmarks)
    for hand_idx in range(2):
        if hand_idx < len(hand_landmarks_list):
            for lm in hand_landmarks_list[hand_idx].landmark:
                row.extend([lm.x, lm.y, lm.z])
        else:
            # Pad with zeros for missing hand
            row.extend([0.0] * 63)  # 21 landmarks × 3 axes
    row.append(label)
    with open(DATA_FILE, 'a', newline='') as f:
        csv.writer(f).writerow(row)

def main():
    init_csv()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera connection.")
        sys.exit(1)

    current_gesture = None
    count           = 0
    collecting      = False

    print("\n╔══════════════════════════════════╗")
    print("║  ISL Data Collection Tool        ║")
    print("╠══════════════════════════════════╣")
    for key, name in GESTURES.items():
        print(f"║  Press [{key}]  →  Collect '{name}'")
    print("║  Press [q]  →  Quit              ║")
    print("╚══════════════════════════════════╝\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Save all detected hands together
            if collecting and count < SAMPLES_PER_GESTURE:
                save_landmarks(result.multi_hand_landmarks, current_gesture)
                count += 1

        # Status overlay
        if collecting:
            gesture_name = GESTURES.get(current_gesture, '---')
            status_text  = f"Collecting: {gesture_name}  [{count}/{SAMPLES_PER_GESTURE}]"
            color        = (0, 200, 0)
        else:
            status_text  = "Press a key (a-h) to start collecting"
            color        = (0, 165, 255)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
        cv2.putText(frame, status_text, (10, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if collecting and count >= SAMPLES_PER_GESTURE:
            collecting = False
            print(f"[✓] Done! Collected {SAMPLES_PER_GESTURE} samples for '{GESTURES[current_gesture]}'")

        cv2.imshow('ISL Data Collection', frame)

        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('q'):
            break
        elif chr(key_press) in GESTURES:
            current_gesture = chr(key_press)
            count           = 0
            collecting      = True
            print(f"[→] Collecting '{GESTURES[current_gesture]}'... hold your hand sign steady.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Data saved to {DATA_FILE}")

if __name__ == '__main__':
    main()
