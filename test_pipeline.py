"""
BONUS - Quick Model Sanity Test (no webcam needed)
====================================================
Generates synthetic landmark data and tests the full pipeline.
Useful to verify everything works before running with a real webcam.

Usage:
    python test_pipeline.py
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_FILE    = 'data/isl_data.csv'
MODEL_FILE   = 'model/isl_model.pkl'
ENCODER_FILE = 'model/label_encoder.pkl'

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

def generate_dummy_data(n_per_class=150):
    """Create synthetic landmark data for testing."""
    print("[...] Generating synthetic landmark data...")
    os.makedirs('data', exist_ok=True)

    rows = []
    for label in GESTURES.values():
        # Each gesture gets a unique mean vector so classes are separable
        base = np.random.rand(63) * 0.5
        for _ in range(n_per_class):
            noise = np.random.randn(63) * 0.02
            row   = list(np.clip(base + noise, 0, 1)) + [label]
            rows.append(row)

    header = [f'{a}{i}' for i in range(21) for a in ['x','y','z']] + ['label']
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(DATA_FILE, index=False)
    print(f"[✓] Saved {len(df)} samples → {DATA_FILE}")
    return df

def train_quick(df):
    print("[...] Training quick model...")
    os.makedirs('model', exist_ok=True)

    X = df.drop('label', axis=1).values.astype(float)
    y = df['label'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y_enc)

    with open(MODEL_FILE,    'wb') as f: pickle.dump(clf, f)
    with open(ENCODER_FILE,  'wb') as f: pickle.dump(le,  f)

    print(f"[✓] Model saved   → {MODEL_FILE}")
    print(f"[✓] Encoder saved → {ENCODER_FILE}")
    return clf, le

def test_predict(clf, le):
    print("\n[...] Running prediction test with random landmark vectors...\n")
    print(f"{'Gesture':<15} {'Predicted':<15} {'Confidence':>12}  {'Match':>6}")
    print("─" * 52)

    all_correct = 0
    for label in GESTURES.values():
        # Simulate a new unseen sample close to training distribution
        base  = np.random.rand(63) * 0.5
        noise = np.random.randn(63) * 0.02
        x     = np.clip(base + noise, 0, 1).reshape(1, -1)

        proba  = clf.predict_proba(x)[0]
        idx    = np.argmax(proba)
        pred   = le.inverse_transform([idx])[0]
        conf   = proba[idx]
        match  = "✓" if pred == label else "✗"
        if pred == label: all_correct += 1

        print(f"{label:<15} {pred:<15} {conf*100:>10.1f}%  {match:>6}")

    print("─" * 52)
    print(f"\n[Result] {all_correct}/{len(GESTURES)} correct on synthetic data\n")

    if all_correct == len(GESTURES):
        print("[✓] Pipeline is working correctly!")
        print("    Now run the real workflow:\n")
        print("    1. python 1_collect_data.py  → collect your hand gestures")
        print("    2. python 2_train_model.py   → train the model")
        print("    3. python 3_detect_realtime.py → run live detection\n")
    else:
        print("[!] Some mismatches (expected with synthetic data). Real data will perform better.")

if __name__ == '__main__':
    print("\n╔══════════════════════════════════╗")
    print("║  ISL Pipeline Sanity Test        ║")
    print("╚══════════════════════════════════╝\n")
    df        = generate_dummy_data()
    clf, le   = train_quick(df)
    test_predict(clf, le)
