import cv2
import app
import numpy as np

# Download a sample traffic image for testing or use your own
test_image_path = "test_traffic.jpg"

# For this test, let's create a simple test image
# In real scenario, you'd use your uploaded image
print("Testing YOLO detection fix...")
print(f"YOLO Available: {app.YOLO_AVAILABLE}")
print(f"YOLO Network: {app.YOLO_NET}")
print(f"Number of classes: {len(app.YOLO_CLASSES)}")

# Create a test image (640x480)
test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Try detection
result = app.detect_cars_and_people(test_img)

if result is not None:
    car_boxes, people_boxes = result
    print(f"\nDetection Results:")
    print(f"  Cars detected: {len(car_boxes)}")
    print(f"  People detected: {len(people_boxes)}")
    if car_boxes:
        print(f"  First car box: {car_boxes[0]}")
else:
    print("Detection returned None")
