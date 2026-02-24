import cv2
import numpy as np

# Simulate the issue
YOLO_INPUT_SIZE = 416
test_img = np.zeros((480, 640, 3), dtype=np.uint8)  # Sample VGA image
h, w = test_img.shape[:2]

# Simulate detection output (center_x, center_y, width, height are in [0, 416] range)
detection = np.array([208, 240, 80, 100, 0.8, 0.1, 0.05])  # center at 208,240 in 416x416

# Current (WRONG) code
center_x = int(detection[0] * w)  # 208 * 640 = 133120 (!!! way off)
center_y = int(detection[1] * h)  # 240 * 480 = 115200 (!!! way off)

print(f"Original image: {w}x{h}")
print(f"YOLO network size: {YOLO_INPUT_SIZE}x{YOLO_INPUT_SIZE}")
print(f"CURRENT (WRONG) calculation:")
print(f"  center_x: {center_x} (should be around {int(208*w/YOLO_INPUT_SIZE)})")
print(f"  center_y: {center_y} (should be around {int(240*h/YOLO_INPUT_SIZE)})")

# Correct calculation
center_x_correct = int(detection[0] * w / YOLO_INPUT_SIZE)
center_y_correct = int(detection[1] * h / YOLO_INPUT_SIZE)
width_correct = int(detection[2] * w / YOLO_INPUT_SIZE)
height_correct = int(detection[3] * h / YOLO_INPUT_SIZE)

print(f"\nCORRECT calculation:")
print(f"  center_x: {center_x_correct}")
print(f"  center_y: {center_y_correct}")
print(f"  width: {width_correct}, height: {height_correct}")
