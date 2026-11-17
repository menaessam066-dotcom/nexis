"""Feature extraction for a fixed, cropped license-plate image.
Demo recognition is for one known plate, NOT robust full-image ANPR.
"""
from __future__ import annotations

import cv2
import numpy as np

WIDTH, HEIGHT = 320, 160
_hog = cv2.HOGDescriptor(
    _winSize=(WIDTH, HEIGHT),
    _blockSize=(32, 32),
    _blockStride=(16, 16),
    _cellSize=(16, 16),
    _nbins=9,
)


def decode_image(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The file is not a valid image.")
    if img.shape[0] < 50 or img.shape[1] < 80:
        raise ValueError("Image is too small; please crop the whole plate clearly.")
    return img


def plate_features(img: np.ndarray) -> np.ndarray:
    # The supplied image should already be tightly cropped to the PLATE.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    feature = _hog.compute(gray).reshape(-1).astype(np.float32)
    # Keep a stable feature magnitude across photos.
    feature /= (float(np.linalg.norm(feature)) + 1e-8)
    return feature
