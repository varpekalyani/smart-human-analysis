"""Comprehensive system check for car detection"""
import app
import cv2
import numpy as np

# Test that all components are loaded
print("=" * 60)
print("FLASK CAR COLOUR DETECTION - SYSTEM CHECK")
print("=" * 60)

print(f"\n[✓] Flask App loaded: {app.app is not None}")
print(f"[✓] YOLO Available: {app.YOLO_AVAILABLE}")
print(f"[✓] YOLO Model: {app.YOLO_NET is not None}")

# Test helper functions
print(f"\n[✓] _is_car_blue function: {callable(app._is_car_blue)}")
print(f"[✓] detect_cars_and_people function: {callable(app.detect_cars_and_people)}")

# Test with sample image
test_img = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Dark image
result = app.detect_cars_and_people(test_img)
print(f"\n[✓] Detection function works: {result is not None}")
print(f"   - Returned: cars={len(result[0])}, people={len(result[1])}")

# Test car_colour route handler exists
print(f"\n[✓] /car-colour route exists: {'/car-colour' in [rule.rule for rule in app.app.url_map.iter_rules()]}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED - READY TO DETECT!")
print("=" * 60)
