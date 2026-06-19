"""
STEP 3 - Real-time ISL Detection
==================================
Live webcam-based ISL gesture detection.
Features:
  - MediaPipe hand landmark extraction
  - Prediction smoothing (majority vote over last N frames)
  - Confidence bar UI
  - Prediction history log

Usage:
    python 3_detect_realtime.py

Controls:
    Press 'q' to quit
    Press 'c' to clear history

NOTE: Requires mediapipe==0.10.14
    pip install mediapipe==0.10.14
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import sys
from collections import deque, Counter
import time

# ── Version check ──────────────────────────────────────────────────────────────
if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'hands'):
    print(f"\n[ERROR] mediapipe {mp.__version__} does not support the solutions API.")
    print("[FIX]  Run:  pip install mediapipe==0.10.14")
    print("       Then re-run this script.\n")
    sys.exit(1)

MODEL_FILE   = 'model/isl_model.pkl'
ENCODER_FILE = 'model/label_encoder.pkl'

LETTER_TO_GESTURE = {
    'a': 'Hello',
    'b': 'Thank You',
    'c': 'Yes',
    'd': 'No',
    'e': 'Please',
    'f': 'Help',
    'g': 'I Love You',
    'h': 'Good',
}

GESTURE_EMOJI = {
    'Hello':       'WAVE',
    'Thank You':   'FOLDED HANDS',
    'Yes':         'THUMBS UP',
    'No':          'PALM STOP',
    'Please':      'HANDS TOGETHER',
    'Help':        'SOS',
    'I Love You':  'ILY SIGN',
    'Good':        'OK SIGN',
}

SMOOTHING_WINDOW  = 10
CONFIDENCE_THRESH = 0.55

# ── Colors (BGR) ───────────────────────────────────────────────────────────────
C_GREEN  = (50,  205, 50)
C_ORANGE = (0,   165, 255)
C_WHITE  = (240, 240, 240)
C_GRAY   = (120, 120, 120)
C_TEAL   = (200, 180, 0)
C_DARK   = (20,  20,  20)

def generate_sentence(gestures):
    """Generate a natural sentence from a list of detected gestures."""
    if not gestures:
        return ""
    
    # Simple rule-based sentence generation
    sentence_templates = {
        ('Hello',): "Hello",
        ('Thank You',): "Thank you",
        ('Yes',): "Yes",
        ('No',): "No",
        ('Please',): "Please",
        ('Help',): "Help me",
        ('I Love You',): "I love you",
        ('Good',): "Good",
        # Multi-gesture combinations
        ('Hello', 'I Love You'): "Hello, I love you",
        ('Please', 'Help'): "Please help me",
        ('Thank You', 'Good'): "Thank you, that's good",
        ('Hello', 'Thank You'): "Hello, thank you",
        ('Yes', 'Good'): "Yes, that's good",
        ('No', 'Thank You'): "No, but thank you",
    }
    
    # Try exact match first
    gestures_tuple = tuple(gestures[-3:]) if len(gestures) > 3 else tuple(gestures)
    if gestures_tuple in sentence_templates:
        return sentence_templates[gestures_tuple]
    
    # Fallback: join gestures with commas
    return ", ".join(gestures[-5:])  # Show last 5 gestures

def load_model():
    try:
        with open(MODEL_FILE,   'rb') as f: model = pickle.load(f)
        with open(ENCODER_FILE, 'rb') as f: le    = pickle.load(f)
        print(f"[✓] Model loaded | Classes: {list(le.classes_)}")
        return model, le
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[FIX]  Run 2_train_model.py first!")
        return None, None

def draw_confidence_bar(frame, confidence, x, y, w=200, h=18):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill_w = int(w * confidence)
    color  = C_GREEN if confidence >= CONFIDENCE_THRESH else C_ORANGE
    cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_GRAY, 1)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (x + w + 8, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_WHITE, 1)

def draw_panel(frame, prediction, confidence, history, fps, sentence=""):
    h, w     = frame.shape[:2]
    panel_x  = w - 265

    # Panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x - 10, 0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "ISL DETECTOR", (panel_x, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, C_TEAL, 1)
    cv2.line(frame, (panel_x - 10, 45), (w, 45), C_GRAY, 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (panel_x, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_GRAY, 1)

    # Sentence generated
    cv2.putText(frame, "SENTENCE", (panel_x, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)
    
    if sentence:
        # Wrap long sentences
        words = sentence.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > 28:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line += (" " if current_line else "") + word
        if current_line:
            lines.append(current_line)
        
        y_offset = 130
        for line in lines[:2]:  # Show up to 2 lines
            cv2.putText(frame, line, (panel_x, y_offset),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, C_GREEN, 2)
            y_offset += 25
    else:
        cv2.putText(frame, "---", (panel_x, 140),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, C_GRAY, 2)

    # Current gesture
    cv2.putText(frame, "CURRENT", (panel_x, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)

    if prediction:
        cv2.putText(frame, prediction, (panel_x, 215),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, C_GREEN, 2)
        desc = GESTURE_EMOJI.get(prediction, '')
        cv2.putText(frame, f"[{desc}]", (panel_x, 238),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_ORANGE, 1)
        cv2.putText(frame, "Confidence", (panel_x, 258),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)
        draw_confidence_bar(frame, confidence, panel_x, 264)
    else:
        cv2.putText(frame, "---", (panel_x, 215),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, C_GRAY, 2)
        cv2.putText(frame, "Show hand to camera", (panel_x, 238),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_ORANGE, 1)

    # History
    cv2.line(frame, (panel_x - 10, 310), (w, 310), C_GRAY, 1)
    cv2.putText(frame, "HISTORY", (panel_x, 332),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_GRAY, 1)

    recent = list(history)[-6:]
    for i, (gest, conf) in enumerate(recent):
        y_pos = 355 + i * 27
        color = C_WHITE if i == len(recent) - 1 else C_GRAY
        cv2.putText(frame, f"- {gest} ({conf*100:.0f}%)", (panel_x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(frame, "q:quit  c:clear", (panel_x, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, C_GRAY, 1)

def main():
    model, le = load_model()
    if model is None:
        sys.exit(1)

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    smooth_buffer = deque(maxlen=SMOOTHING_WINDOW)
    history       = deque(maxlen=20)
    detected_gestures = deque(maxlen=10)  # Track gestures for sentence generation
    last_pred     = None
    last_conf     = 0.0
    prev_time     = time.time()
    fps           = 0.0

    print("\n[►] Real-time ISL Detection started.")
    print("    Press 'q' to quit | 'c' to clear history\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style()
                )

                features = []
                for lm in hand_lms.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                features    = np.array(features).reshape(1, -1)
                proba       = model.predict_proba(features)[0]
                pred_idx    = np.argmax(proba)
                confidence  = float(proba[pred_idx])
                pred_label  = le.inverse_transform([pred_idx])[0]
                pred_gesture = LETTER_TO_GESTURE.get(pred_label, pred_label)

                smooth_buffer.append(pred_gesture)
                smoothed = Counter(smooth_buffer).most_common(1)[0][0]

                if confidence >= CONFIDENCE_THRESH:
                    last_pred = smoothed
                    last_conf = confidence
                    if not history or history[-1][0] != smoothed:
                        history.append((smoothed, confidence))
                        detected_gestures.append(smoothed)
        else:
            cv2.putText(frame, "Show your hand to the camera...",
                        (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_ORANGE, 2)

        # Generate sentence from detected gestures
        sentence = generate_sentence(list(detected_gestures))
        draw_panel(frame, last_pred, last_conf, history, fps, sentence)
        cv2.imshow('ISL Real-time Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            history.clear()
            detected_gestures.clear()
            last_pred = None
            last_conf = 0.0
            print("[INFO] History cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Detection stopped.")

if __name__ == '__main__':
    main()
