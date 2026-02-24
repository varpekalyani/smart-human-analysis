#!/usr/bin/env python3
"""
Diagnostic script to verify model loading and environment for the
Smart Human Object Analysis System (Long Hair Identification).

Run: python check_models.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("=" * 60)
print("Smart Human Object Analysis System - Model Diagnostics")
print("=" * 60)

# 1. Python version
print(f"\n[1] Python: {sys.version}")

# 2. TensorFlow (optional - app uses OpenCV DNN, not TF for age/gender)
print("\n[2] TensorFlow / Keras:")
try:
    import tensorflow as tf
    print(f"    TensorFlow: {tf.__version__} (OK)")
    try:
        import keras
        print(f"    Keras:      {keras.__version__} (OK)")
    except ImportError:
        print("    Keras:      not installed (optional)")
except ImportError as e:
    print(f"    TensorFlow: not installed (optional - app uses OpenCV DNN)")

# 3. OpenCV and DNN
print("\n[3] OpenCV:")
try:
    import cv2
    print(f"    Version: {cv2.__version__}")
    if hasattr(cv2.dnn, "readNetFromCaffe"):
        print("    DNN Caffe support: OK")
    else:
        print("    DNN Caffe support: MISSING (models may not load)")
except ImportError as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

# 4. Model files
print("\n[4] Model files:")
expected = [
    ("opencv_face_detector.pbtxt", "Face DNN config"),
    ("opencv_face_detector_uint8.pb", "Face DNN weights"),
    ("age_deploy.prototxt", "Age Caffe config"),
    ("age_net.caffemodel", "Age Caffe weights"),
    ("gender_deploy.prototxt", "Gender Caffe config"),
    ("gender_net.caffemodel", "Gender Caffe weights"),
]
all_ok = True
for fname, desc in expected:
    path = os.path.join(MODELS_DIR, fname)
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    status = "OK" if exists and size > 0 else "MISSING or empty"
    if not (exists and size > 0):
        all_ok = False
    print(f"    {fname}: {status} ({desc})")

# 5. Try loading models
print("\n[5] Loading models:")
try:
    import cv2
    import numpy as np

    # Face (TensorFlow)
    face_proto = os.path.join(MODELS_DIR, "opencv_face_detector.pbtxt")
    face_model = os.path.join(MODELS_DIR, "opencv_face_detector_uint8.pb")
    try:
        face_net = cv2.dnn.readNet(face_model, face_proto)
        print(f"    Face:  {'OK' if not face_net.empty() else 'FAILED (empty net)'}")
    except Exception as e:
        print(f"    Face:  FAILED - {e}")

    # Age (Caffe - prototxt first, caffemodel second)
    age_proto = os.path.join(MODELS_DIR, "age_deploy.prototxt")
    age_model = os.path.join(MODELS_DIR, "age_net.caffemodel")
    try:
        age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
        print(f"    Age:   {'OK' if not age_net.empty() else 'FAILED (empty net)'}")
    except Exception as e:
        print(f"    Age:   FAILED - {e}")

    # Gender (Caffe)
    gender_proto = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
    gender_model = os.path.join(MODELS_DIR, "gender_net.caffemodel")
    try:
        gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
        print(f"    Gender:{'OK' if not gender_net.empty() else 'FAILED (empty net)'}")
    except Exception as e:
        print(f"    Gender:FAILED - {e}")

except Exception as e:
    print(f"    Error during load: {e}")

print("\n" + "=" * 60)
print("If any model shows MISSING or FAILED:")
print("  - Ensure models are in:", MODELS_DIR)
print("  - Age/Gender models: Caffe format (prototxt + caffemodel)")
print("  - Download from: https://github.com/arunponnusamy/cvlib")
print("    or search 'opencv age gender caffe model'")
print("=" * 60)
