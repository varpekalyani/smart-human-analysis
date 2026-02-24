#!/usr/bin/env python3
"""Download Face, Age, Gender models for Long Hair Identification."""
import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = [
    ("opencv_face_detector.pbtxt",
     "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"),
    ("opencv_face_detector_uint8.pb",
     "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb"),
    ("age_deploy.prototxt",
     "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"),
    ("age_net.caffemodel",
     "https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1"),
    ("gender_deploy.prototxt",
     "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"),
    ("gender_net.caffemodel",
     "https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=1"),
]

def main():
    print("Downloading Face, Age, Gender models...")
    for fname, url in MODELS:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            print(f"  [SKIP] {fname} (exists)")
            continue
        print(f"  [DOWN] {fname}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, path)
            print(f"OK ({os.path.getsize(path)} bytes)")
        except Exception as e:
            print(f"FAILED: {e}")
            sys.exit(1)
    print("Done. Run: python check_models.py")

if __name__ == "__main__":
    main()
