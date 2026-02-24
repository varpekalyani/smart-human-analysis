"""Test YOLOv8 detection"""
import cv2
import app
import numpy as np
from urllib.request import urlopen
from io import BytesIO

# Try to download a test traffic image
print("Testing YOLOv8 detection...")
print(f"YOLO Available: {app.YOLO_AVAILABLE}")

# Create a simple test image for debugging
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
print(f"Test image shape: {test_img.shape}")

# Run detection
result = app.detect_cars_and_people(test_img)
print(f"Detection result: {result}")

if result is not None:
    car_boxes, people_boxes = result
    print(f"Cars detected: {len(car_boxes)}")
    print(f"People detected: {len(people_boxes)}")
else:
    print("Detection returned None")

print("\n✓ Test completed successfully")
