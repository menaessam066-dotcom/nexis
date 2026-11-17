"""Train binary classifier for ONE cropped target plate (not 50 separate IDs).

Split data before fitting: train 70%, threshold calibration 15%, untouched test 15%.
Do not use training or calibration metrics as proof of real-camera accuracy.
"""
from __future__ import annotations
from pathlib import Path
import json
import hashlib
import os
import tempfile
import argparse
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from plate_core import plate_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'plate_model.joblib'
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_data():
    feats, labels, names = [], [], []
    hashes = {}
    for label, folder in [(1, ROOT / 'data' / 'target'), (0, ROOT / 'data' / 'other')]:
        entries = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if len(entries) < 10:
            raise RuntimeError(f'Need >=10 readable images in {folder}; found {len(entries)} files.')
        for p in entries:
            blob = p.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if digest in hashes:
                print(f'WARNING duplicate exact file: {p} matches {hashes[digest]}')
                if label != hashes[digest][0]:
                    raise RuntimeError('The SAME image exists in target and other. Correct labels before training.')
            else:
                hashes[digest] = (label, p)
            arr = np.frombuffer(blob, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None or min(img.shape[:2]) < 50:
                print(f'SKIP invalid or too small: {p}')
                continue
            feats.append(plate_features(img))
            labels.append(label)
            names.append(str(p.relative_to(ROOT)))
    X, y = np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int32)
    if len(X) < 30 or len(np.unique(y)) != 2 or min(np.bincount(y)) < 10:
        raise RuntimeError('Need >=10 readable photos of each class and >=30 total.')
    return X, y, names


def fit(X, y):
    model = SVC(kernel='rbf', C=4.0, gamma='scale', class_weight='balanced')
    model.fit(X, y)
    return model


def choose_threshold(scores, truth):
    """Choose high-specificity threshold using calibration ONLY, never test images."""
    vals = np.unique(scores)
    if len(vals) == 1:
        return float(vals[0]), 'Calibrated scores identical; collect more diverse data.'
    candidates = np.r_[vals[0] - .001, (vals[:-1] + vals[1:]) / 2, vals[-1] + .001]
    rated = []
    for t in candidates:
        pred = scores >= t
        fp = int(np.sum(pred & (truth == 0)))
        fn = int(np.sum(~pred & (truth == 1)))
        tp = int(np.sum(pred & (truth == 1)))
        rated.append((fp, fn, -tp, float(t)))
    # First avoid opening for known negatives, then maximize matching true positives.
    best = min(rated)
    threshold = best[3]
    warn = ('Zero false accepts on calibration; this does NOT guarantee zero false accepts on new real plates.'
            if best[0] == 0 else 'Some calibration negatives were accepted; increase data variety and review labels.')
    if best[2] == 0:
        warn += ' WARNING: threshold accepts zero target calibration examples.'
    return threshold, warn


def train(real_data: bool = False):
    print('Loading cropped images from data/target and data/other...')
    X, y, names = load_data()
    all_idx = np.arange(len(y))
    # Separate final test set first, THEN calibration set from remaining images.
    development_idx, test_idx = train_test_split(all_idx, test_size=.15, random_state=42, stratify=y)
    training_idx, calibration_idx = train_test_split(development_idx, test_size=.1764705882, random_state=81, stratify=y[development_idx])
    model = fit(X[training_idx], y[training_idx])
    calibration_scores = model.decision_function(X[calibration_idx])
    threshold, threshold_note = choose_threshold(calibration_scores, y[calibration_idx])
    test_scores = model.decision_function(X[test_idx])
    predictions = (test_scores >= threshold).astype(np.int32)
    cm = confusion_matrix(y[test_idx], predictions, labels=[0,1])
    report = {
        'algorithm': 'HOG + RBF Support Vector Machine',
        'scope': 'ONE target license plate (all target photos share the exact plate ID); CROPPED plates only',
        'dataset_type': ('User-supplied photos (not independently verified)' if real_data else 'Synthetic demonstration data'),
        'image_count_target': int(np.sum(y == 1)),
        'image_count_other': int(np.sum(y == 0)),
        'train_count': int(len(training_idx)),
        'calibration_count': int(len(calibration_idx)),
        'test_count': int(len(test_idx)),
        'validation_count': int(len(test_idx)),
        'validation_accuracy': round(float(accuracy_score(y[test_idx], predictions)),4),
        'validation_precision': round(float(precision_score(y[test_idx], predictions, zero_division=0)),4),
        'validation_recall': round(float(recall_score(y[test_idx], predictions, zero_division=0)),4),
        'confusion_matrix': {'TN': int(cm[0,0]), 'FP': int(cm[0,1]), 'FN': int(cm[1,0]), 'TP': int(cm[1,1])},
        'decision_threshold': round(float(threshold),6),
        'threshold_note': threshold_note,
        'warning': 'Random splits of synthetic images are optimistic. Never assume these scores generalize to real plates. A separate real photo test is required. Not security-grade.',
        'evaluation_protocol': 'Train ~70%, calibrate threshold ~15%, evaluate untouched test ~15%; model saved without training on test.',
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement so the running dashboard never sees a half-written model.
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.joblib', dir=MODEL_PATH.parent, delete=False) as tmp:
        path = Path(tmp.name)
    try:
        joblib.dump({'model': model, 'threshold': threshold, 'report': report}, path)
        os.replace(path, MODEL_PATH)
    finally:
        path.unlink(missing_ok=True)
    (ROOT / 'models' / 'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nModel saved:', MODEL_PATH)
    print(json.dumps(report, indent=2))
    print('\nIMPORTANT: Test with brand-new cropped photos. Training metrics alone are not enough.')
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-data', action='store_true', help='Mark the retraining report as user-supplied real photos; use only after replacing BOTH training folders')
    args = parser.parse_args()
    train(real_data=args.real_data)
