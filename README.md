# 🤟 ISL Sign Language Detection
**Real-time Indian Sign Language detection using MediaPipe + Random Forest**

---

## 📁 Project Structure

```
isl_detection/
├── 1_collect_data.py      # Step 1 - Collect hand landmark data via webcam
├── 2_train_model.py       # Step 2 - Train Random Forest classifier
├── 3_detect_realtime.py   # Step 3 - Live webcam detection
├── test_pipeline.py       # Bonus  - Test the pipeline without webcam
├── requirements.txt
├── data/
│   └── isl_data.csv       # Generated after Step 1
└── model/
    ├── isl_model.pkl       # Generated after Step 2
    └── label_encoder.pkl   # Generated after Step 2
```

---

## 🚀 Setup

```bash
pip install -r requirements.txt
```

---

## 🔢 How It Works

```
Webcam Frame
    ↓
MediaPipe Hands (up to 2 hands)
    ↓
42 Landmarks (x, y, z) = 126 features
(21 landmarks × 3 axes × 2 hands, with padding for 1-hand signs)
    ↓
Random Forest Classifier
    ↓
ISL Gesture Label + Confidence
    ↓
Sentence Generation
```

---

## ▶️ Run Order

### Step 1 — Collect Data
```bash
python 1_collect_data.py
```
- Press `a` → collect "Hello"
- Press `b` → collect "Thank You"
- Press `c` → collect "Yes"
- Press `d` → collect "No"
- Press `e` → collect "Please"
- Press `f` → collect "Help"
- Press `g` → collect "I Love You"
- Press `h` → collect "Good"
- Press `q` → quit

Collect **200 samples per gesture** (hold your hand sign in front of camera).

### Step 2 — Train Model
```bash
python 2_train_model.py
```
Trains a Random Forest on your collected data. Prints accuracy + classification report.

### Step 3 — Live Detection
```bash
python 3_detect_realtime.py
```
- Shows live webcam with hand skeleton
- Right panel shows: current prediction, confidence bar, history log
- Press `q` → quit | `c` → clear history

---

## 🧪 Quick Test (No Webcam)
```bash
python test_pipeline.py
```
Generates synthetic data, trains, and tests the full pipeline.

---

## ✏️ Add More Gestures
Edit the `GESTURES` dict in both `1_collect_data.py` and `3_detect_realtime.py`:
```python
GESTURES = {
    'a': 'Hello',
    'i': 'Water',    # add new ones like this
    'j': 'Eat',
}
```

---

## 📊 Tech Stack
| Component | Library |
|-----------|---------|
| Hand Detection | MediaPipe |
| Feature Extraction | 21 Landmarks × (x,y,z) |
| Classifier | Scikit-learn Random Forest |
| Webcam / Drawing | OpenCV |
| Data | CSV via Pandas |

---

## 💡 Tips for Better Accuracy
- Collect data in **good lighting**
- Vary your hand position slightly while collecting (don't stay 100% still)
- Collect at least **200 samples per gesture**
- Sit at a consistent **distance from camera**
- Retrain after adding new gestures

---

*Built for AIML placement portfolio — ISL Detection Mini Project*
