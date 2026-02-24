Project Name: Smart Human Object Analysis System

Description:
A unified AI-based system containing multiple modules such as
senior citizen detection, long hair-based gender identification,
voice-based age and emotion detection, sign language recognition,
car colour detection, and nationality analysis.

Structure:
Single dashboard with independent task-based modules.

Setup:
1. Create and activate a virtual environment (recommended).
2. Install dependencies from requirements.txt:
   pip install -r requirements.txt

   Or install individually:
   pip install flask opencv-python numpy pandas librosa

3. Ensure pandas is installed (required for senior log CSV):
   pip install pandas

4. Place DNN model files in the models/ directory for full functionality:
   - opencv_face_detector_uint8.pb + opencv_face_detector.pbtxt
   - age_net.caffemodel + age_deploy.prototxt
   - gender_net.caffemodel + gender_deploy.prototxt
   - yolov4.weights + yolov4.cfg + coco.names (for car colour detection)

   For Car Colour Detection (YOLOv4-tiny recommended):
   - yolov4-tiny.cfg and coco.names are included.
   - Download yolov4-tiny.weights (~6 MB): .\download_yolov4_tiny.ps1
   - Or full YOLOv4 (~162 MB): .\download_yolov4.ps1

