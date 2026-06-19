"""
STEP 2 - ISL Model Training
==============================
Trains a Random Forest classifier on the collected landmark data.
Saves the model + label encoder to model/ directory.

Usage:
    python 2_train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

DATA_FILE   = 'data/isl_data.csv'
MODEL_FILE  = 'model/isl_model.pkl'
ENCODER_FILE = 'model/label_encoder.pkl'

def train():
    print("\n╔══════════════════════════════════╗")
    print("║  ISL Model Training              ║")
    print("╚══════════════════════════════════╝\n")

    # ── Load Data ──────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] {DATA_FILE} not found. Run 1_collect_data.py first!")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"[INFO] Dataset shape : {df.shape}")
    print(f"[INFO] Gesture counts:\n{df['label'].value_counts()}\n")

    X = df.drop('label', axis=1).values
    y = df['label'].values

    # ── Encode Labels ──────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"[INFO] Classes : {list(le.classes_)}")

    # ── Train / Test Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"[INFO] Train : {len(X_train)} | Test : {len(X_test)}\n")

    # ── Random Forest ──────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    print("[...] Training Random Forest...")
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n[✓] Accuracy : {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Save Model ─────────────────────────────────────────────────────────────
    os.makedirs('model', exist_ok=True)
    with open(MODEL_FILE,   'wb') as f: pickle.dump(model, f)
    with open(ENCODER_FILE, 'wb') as f: pickle.dump(le, f)

    print(f"[✓] Model saved   → {MODEL_FILE}")
    print(f"[✓] Encoder saved → {ENCODER_FILE}")

    # ── Feature Importance (top 10) ────────────────────────────────────────────
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    print("\n[INFO] Top 10 important landmarks:")
    for idx in top_idx:
        lm_num  = idx // 3
        coord   = ['x','y','z'][idx % 3]
        print(f"   Landmark {lm_num:02d} [{coord}]  →  importance: {importances[idx]:.4f}")

if __name__ == '__main__':
    train()
