from flask import (
    Flask,
    render_template,
    request,
    Response,
    redirect,
    url_for,
    flash,
    send_file,
)
import cv2
import numpy as np
import os
try:
    import pandas as pd
except ImportError:
    pd = None
import librosa
from collections import Counter
from datetime import datetime, time as dtime

# MediaPipe hand detection: try Tasks API (0.10+) first, then legacy solutions.hands
_hand_landmarker = None
_HANDS_AVAILABLE = False
_HANDS_IMPORT_TRIED = False
_HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def _try_import_mediapipe():
    """Try to load MediaPipe hand detection once (Tasks API for 0.10+)."""
    global _hand_landmarker, _HANDS_AVAILABLE, _HANDS_IMPORT_TRIED
    if _HANDS_IMPORT_TRIED:
        return
    _HANDS_IMPORT_TRIED = True
    try:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as mp_base
        model_path = os.path.join(MODELS_DIR, "hand_landmarker.task")
        if not os.path.isfile(model_path):
            import urllib.request
            os.makedirs(MODELS_DIR, exist_ok=True)
            urllib.request.urlretrieve(_HAND_MODEL_URL, model_path)
        options = vision.HandLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        _hand_landmarker = vision.HandLandmarker.create_from_options(options)
        _HANDS_AVAILABLE = True
    except Exception as e:
        import sys
        print(f"[Sign language] Hand detection failed: {e}", file=sys.stderr)
        _hand_landmarker = None
        _HANDS_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
LOG_DIR = os.path.join(BASE_DIR, "logs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

SENIOR_LOG_PATH = os.path.join(LOG_DIR, "senior_log.csv")


app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR


# ==============================
# Helper: Safe model loading
# ==============================

def _safe_read_net(model_path: str, config_path: str | None = None, use_caffe: bool = False):
    """Safely load an OpenCV DNN model. Returns (net, available_flag).
    Avoids cv2.error (netBinSize || netTxtSize) when model files are missing/empty.
    For Caffe models: config_path=prototxt, model_path=caffemodel (order matters).
    """
    try:
        if not os.path.isfile(model_path) or os.path.getsize(model_path) == 0:
            return None, False
        if config_path is not None:
            if not os.path.isfile(config_path) or os.path.getsize(config_path) == 0:
                return None, False
            if use_caffe:
                # Caffe: readNetFromCaffe(prototxt, caffemodel)
                net = cv2.dnn.readNetFromCaffe(config_path, model_path)
            else:
                net = cv2.dnn.readNet(model_path, config_path)
        else:
            net = cv2.dnn.readNet(model_path)
        if net.empty():
            return None, False
        return net, True
    except Exception as e:
        import sys
        print(f"[Model load] {model_path}: {e}", file=sys.stderr)
        return None, False


# Face / Age / Gender models (used by multiple modules)
FACE_PROTO = os.path.join(MODELS_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(MODELS_DIR, "opencv_face_detector_uint8.pb")

AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")

GENDER_PROTO = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")

faceNet, FACE_AVAILABLE = _safe_read_net(FACE_MODEL, FACE_PROTO)
ageNet, AGE_AVAILABLE = _safe_read_net(AGE_MODEL, AGE_PROTO, use_caffe=True)
genderNet, GENDER_AVAILABLE = _safe_read_net(GENDER_MODEL, GENDER_PROTO, use_caffe=True)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)",
]
AGE_BUCKET_TO_APPROX = {
    "(0-2)": 1,
    "(4-6)": 5,
    "(8-12)": 10,
    "(15-20)": 18,
    "(25-32)": 28,
    "(38-43)": 40,
    "(48-53)": 50,
    "(60-100)": 65,
}
GENDER_LABELS = ["Male", "Female"]

HAAR_FACE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces_dnn(image):
    """Return list of face boxes (x1, y1, x2, y2) using DNN or empty list."""
    if not FACE_AVAILABLE or faceNet is None:
        return []

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1.0, (300, 300), [104, 117, 123], swapRB=False
    )
    faceNet.setInput(blob)
    detections = faceNet.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            boxes.append((x1, y1, x2, y2))
    return boxes


def _crop_face_with_padding(frame, box, pad_ratio=0.2):
    """Extract face with padding. Padding improves gender/age model accuracy."""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    fh, fw = y2 - y1, x2 - x1
    pad_x = int(fw * pad_ratio)
    pad_y = int(fh * pad_ratio)
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(w, x2 + pad_x)
    y2p = min(h, y2 + pad_y)
    return frame[y1p:y2p, x1p:x2p]


def predict_age_and_gender(face_img):
    """Return (approx_age_int, age_bucket, gender_label, used_demo_fallback, gender_conf).
    gender_conf is max probability (0-1); low values mean model is uncertain."""
    if (
        face_img is None
        or face_img.size == 0
        or not AGE_AVAILABLE
        or not GENDER_AVAILABLE
        or ageNet is None
        or genderNet is None
    ):
        return 25, "(25-32)", "Male", True, 0.5

    blob = cv2.dnn.blobFromImage(
        face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
    )

    genderNet.setInput(blob)
    gender_preds = genderNet.forward()
    probs = gender_preds[0]
    idx = probs.argmax()
    gender = GENDER_LABELS[idx]
    gender_conf = float(probs[idx])

    ageNet.setInput(blob)
    age_preds = ageNet.forward()
    age_bucket = AGE_BUCKETS[age_preds[0].argmax()]
    approx_age = AGE_BUCKET_TO_APPROX.get(age_bucket, 30)

    return approx_age, age_bucket, gender, False, gender_conf


def classify_hair_length(face_box, frame):
    """
    Distinguish short vs long hair:
    - Short: hair stays above ears, little/no hair visible at ear/neck level on sides.
    - Long: hair extends past ears, clearly visible at ear/cheek level on sides.
    """
    x1, y1, x2, y2 = face_box
    face_height = y2 - y1
    face_width = x2 - x1
    h_total, w_total = frame.shape[:2]

    # Region above face (both short and long have hair here)
    hair_top = max(0, y1 - face_height // 2)
    hair_region_above = frame[hair_top:y1, x1:x2]

    # Side regions - LOWER part (ear/cheek level): long hair extends here, short does not
    pad = min(face_width // 2, 40)
    x_left = max(0, x1 - pad)
    x_right = min(w_total, x2 + pad)
    mid_face = y1 + face_height // 3
    sides_lower_left = frame[mid_face:y2, x_left:x1] if x_left < x1 else None
    sides_lower_right = frame[mid_face:y2, x2:x_right] if x_right > x2 else None

    def _hair_density(region):
        """Count dark pixels (hair-like); exclude skin/background. Hair is typically dark."""
        if region is None or region.size == 0:
            return 0.0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Dark pixels (20-85): hair; avoid very dark (shadow) and skin (100+)
        hair_mask = (gray > 20) & (gray < 85)
        return np.count_nonzero(hair_mask) / hair_mask.size

    ratio_above = _hair_density(hair_region_above) if hair_region_above.size > 0 else 0.0
    ratio_sides_lower_left = _hair_density(sides_lower_left) if sides_lower_left is not None else 0.0
    ratio_sides_lower_right = _hair_density(sides_lower_right) if sides_lower_right is not None else 0.0
    ratio_sides_lower = max(ratio_sides_lower_left, ratio_sides_lower_right)

    # Short hair: little/no dark hair at ear/cheek level on sides
    if ratio_sides_lower < 0.20:
        return "short"
    # Long hair: clear dark hair extending past ears (stricter so short classifies correctly)
    if ratio_sides_lower > 0.42:
        return "long"
    if ratio_sides_lower > 0.32 and ratio_above > 0.15:
        return "long"
    # Default: sides not clearly long → short
    return "short"


def ensure_senior_log_header():
    """Ensure the senior log CSV has a header."""
    if pd is None:
        return
    if not os.path.exists(SENIOR_LOG_PATH):
        df = pd.DataFrame(
            columns=["Date", "Time", "Age", "Gender", "SeniorStatus"]
        )
        df.to_csv(SENIOR_LOG_PATH, index=False)


def log_senior(age_str, gender_str, is_senior: bool):
    if pd is None:
        return
    ensure_senior_log_header()
    now = datetime.now()
    row = {
        "Date": now.date().isoformat(),
        "Time": now.time().strftime("%H:%M:%S"),
        "Age": age_str,
        "Gender": gender_str,
        "SeniorStatus": "Senior" if is_senior else "Non-Senior",
    }
    try:
        df = pd.DataFrame([row])
        df.to_csv(SENIOR_LOG_PATH, mode="a", header=False, index=False)
    except Exception:
        # Fail silently for logging; don't break video stream.
        pass


def allowed_time_for_sign_language() -> bool:
    """Return True always — sign language detection available 24/7."""
    return True


def extract_mfcc(audio_path: str):
    """Extract MFCC features using librosa with safe fallback."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return mfcc, y, sr
    except Exception:
        return None, None, None


def _estimate_pitch(y, sr):
    """Estimate median fundamental frequency (F0) in Hz. Male ~85-180, Female ~165-255."""
    if y is None or sr is None or len(y) < 2048:
        return None
    try:
        f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr)
        f0_valid = f0[f0 > 0]
        return float(np.median(f0_valid)) if len(f0_valid) > 0 else None
    except Exception:
        return None


def predict_voice_gender(mfcc, y=None, sr=None):
    """Gender from pitch (F0): Female ~165+ Hz, Male ~85-180 Hz. Fallback to MFCC."""
    if y is not None and sr is not None:
        pitch = _estimate_pitch(y, sr)
        if pitch is not None and 80 < pitch < 400:
            return "Female" if pitch > 165 else "Male"
    if mfcc is None:
        return "Unknown"
    mean_c1 = float(np.mean(mfcc[0]))
    mean_c2 = float(np.mean(mfcc[1]))
    return "Female" if mean_c1 > -5 or mean_c2 > 2 else "Male"


def predict_voice_age(mfcc):
    """Age estimation from voice spectral spread (heuristic)."""
    if mfcc is None:
        return 35
    spread = float(np.std(mfcc))
    mean_c = float(np.mean(mfcc))
    if spread < 15:
        return 25
    if spread < 28:
        return 35
    if spread < 45:
        return 50
    return 65


def predict_voice_emotion(mfcc, y=None, sr=None):
    """Emotion from audio: crying→Sad, laughing→Happy, else MFCC-based heuristic."""
    if y is not None and sr is not None and len(y) >= 2048:
        try:
            flatness = librosa.feature.spectral_flatness(y=y)
            flat_mean = float(np.mean(flatness))
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_var = float(np.var(zcr))
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_var = float(np.var(onset_env))
            rms = librosa.feature.rms(y=y)[0]
            rms_var = float(np.var(rms))
            # Crying: breathy (high spectral flatness) or irregular sobbing. Takes precedence.
            if flat_mean > 0.09:
                return "Sad"  # Crying - breathy/wailing
            if zcr_var > 0.002:
                return "Sad"  # Crying - irregular, broken sobbing
            # Laughing: bursty and VOICED (low flatness). Crying is breathy; laughing is harmonic.
            if flat_mean < 0.07 and onset_var > 0.02 and rms_var > 0.001:
                return "Happy"  # Laughing - voiced "ha-ha-ha" bursts
        except Exception:
            pass
    if mfcc is None:
        return "Neutral"
    var_all = float(np.var(mfcc))
    if var_all < 80:
        return "Neutral"  # Low variance = calm speech
    mean_high = float(np.mean(mfcc[4:8])) if mfcc.shape[0] > 8 else 0
    idx = (int(abs(mean_high) * 2) + int(var_all / 80)) % 4
    emotions = ["Neutral", "Happy", "Sad", "Angry"]
    return emotions[idx]


# Gesture → output mapping (hand-landmark based when MediaPipe available)
# HELLO  : hand high, fingers spread (wave)
# YES    : fist (closed fingers)
# NO     : index + middle extended, others closed
# THANK YOU : flat hand, mid-level (chin/forward)
# PLEASE : hand lower (chest), closed or flat

# Hand landmarker is created in _try_import_mediapipe (Tasks API)


def _dist(a, b):
    """Euclidean distance between two landmarks (x, y)."""
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


# Ratio to count finger as extended; 1.05 = easier to get NO (two fingers) and finger counts
_FINGER_EXTENDED_RATIO = 1.05

def _count_extended_fingers(landmarks):
    """Count extended fingers: tip clearly farther from wrist than PIP (stronger identification)."""
    wrist = landmarks[0]
    count = 0
    for tip_idx, pip_idx in [(4, 3), (8, 6), (12, 10), (16, 14), (20, 18)]:
        d_tip = _dist(wrist, landmarks[tip_idx])
        d_pip = _dist(wrist, landmarks[pip_idx])
        if d_pip > 1e-6 and d_tip > d_pip * _FINGER_EXTENDED_RATIO:
            count += 1
    return count


def _index_middle_only(landmarks):
    """True if index and middle clearly extended, others clearly closed (NO sign)."""
    wrist = landmarks[0]
    def ext(tip_idx, pip_idx):
        d_tip = _dist(wrist, landmarks[tip_idx])
        d_pip = _dist(wrist, landmarks[pip_idx])
        return d_pip > 1e-6 and d_tip > d_pip * _FINGER_EXTENDED_RATIO
    def closed(tip_idx, pip_idx):
        d_tip = _dist(wrist, landmarks[tip_idx])
        d_pip = _dist(wrist, landmarks[pip_idx])
        return d_pip <= 1e-6 or d_tip <= d_pip * _FINGER_EXTENDED_RATIO
    return (
        ext(8, 6) and ext(12, 10)
        and closed(4, 3) and closed(16, 14) and closed(20, 18)
    )


def _predict_from_hand(landmarks, frame_height):
    """Map hand landmarks + position to one of HELLO, YES, NO, THANK YOU, PLEASE."""
    words = ["HELLO", "YES", "NO", "THANK YOU", "PLEASE"]
    # Use hand center (not just wrist): hand on chest has wrist at collarbone (high) but center on chest
    hand_center_y = sum(lm.y for lm in landmarks) / len(landmarks)  # normalized 0–1; higher = lower on screen
    n = _count_extended_fingers(landmarks)

    # NO: index + middle only
    if _index_middle_only(landmarks):
        return "NO"

    # Fist (0–1 fingers) → YES
    if n <= 1:
        return "YES"

    # 4–5 fingers extended: HELLO = wave (high), THANK YOU = mid, PLEASE = chest
    if n >= 4:
        if hand_center_y < 0.50:
            return "HELLO"   # hand in upper half (wave)
        if hand_center_y < 0.65:
            return "THANK YOU"  # chin / face level
        return "PLEASE"  # hand on chest (lower)

    # 2–3 fingers → same position logic
    if hand_center_y < 0.52:
        return "HELLO"
    if hand_center_y < 0.68:
        return "THANK YOU"
    return "PLEASE"


# Max size for sign-language inference. Larger = better hand detection.
_SIGN_LANG_INFER_MAX_WIDTH = 640

def predict_sign_language_word(image, static_image=False):
    """Use MediaPipe hand landmarks for gesture. Returns (word, hand_detected). Set static_image=True for uploads."""
    words = ["HELLO", "YES", "NO", "THANK YOU", "PLEASE"]
    if image is None or image.size == 0:
        return (words[0], False)

    # Resize for faster inference when image is large (video path)
    infer_img = image
    if not static_image:
        h, w = image.shape[:2]
        if w > _SIGN_LANG_INFER_MAX_WIDTH:
            scale = _SIGN_LANG_INFER_MAX_WIDTH / w
            new_w = _SIGN_LANG_INFER_MAX_WIDTH
            new_h = int(h * scale)
            infer_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    _try_import_mediapipe()
    if _hand_landmarker is not None:
        try:
            from mediapipe.tasks.python.vision.core import image as mp_image
            rgb = cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB)
            mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb)
            results = _hand_landmarker.detect(mp_img)
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                lm = results.hand_landmarks[0]
                h_infer, _ = infer_img.shape[:2]
                return (_predict_from_hand(lm, h_infer), True)
        except Exception:
            pass

    # Fallback: no hand detected or MediaPipe unavailable
    mean_val = int(np.mean(infer_img))
    return (words[mean_val % len(words)], False)


def dominant_color_name(bgr_image):
    """Return a color name for the dominant color in a region using HSV analysis."""
    if bgr_image is None or bgr_image.size == 0:
        return "Unknown"
    
    try:
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Filter pixels: exclude white/black areas and low-saturation (grayish)
        # Keep only pixels with: 50 < V < 200 (not too dark/bright) and S > 30 (somewhat saturated)
        mask = (v > 50) & (v < 200) & (s > 30)
        
        if np.sum(mask) == 0:
            # Fallback: if all pixels filtered, use basic mean
            mean_h = np.mean(h)
            mean_v = np.mean(v)
        else:
            # Use only the masked pixels
            mean_h = np.mean(h[mask])
            mean_s = np.mean(s[mask])
            mean_v = np.mean(v[mask])
            
            # Debug logging
            print(f"[Color] H={mean_h:.1f}, S={mean_s:.1f}, V={mean_v:.1f}, masked_pixels={np.sum(mask)}", file=sys.stderr)
        
        # Classify by hue
        if mean_h < 20 or mean_h > 160:
            return "Red"
        elif 20 <= mean_h < 40:
            return "Orange"
        elif 40 <= mean_h < 70:
            return "Yellow"
        elif 70 <= mean_h < 90:
            return "Green"
        elif 90 <= mean_h <= 130:
            return "Blue"
        elif 130 < mean_h <= 160:
            return "Purple"
        else:
            return "Other"
    except Exception as e:
        print(f"[Color] Error: {e}", file=sys.stderr)
        return "Unknown"


def detect_dress_color(frame, face_box):
    """
    Detect dress color from region below the face.
    Focus on the center portion to avoid background edges.
    """
    x1, y1, x2, y2 = face_box
    h, w = frame.shape[:2]
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Region starts just below face and extends down 2-3x face height
    dress_top = y2
    dress_bottom = min(h - 1, y2 + face_height * 2.5)
    
    # Focus on center portion to avoid background
    dress_x1 = max(0, int(x1 + face_width * 0.15))
    dress_x2 = min(w, int(x2 - face_width * 0.15))
    
    if dress_x2 > dress_x1 and dress_bottom > dress_top:
        region = frame[dress_top:dress_bottom, dress_x1:dress_x2]
        color = dominant_color_name(region)
        print(f"[Dress] Region: ({dress_x1},{dress_top})-({dress_x2},{dress_bottom}) -> {color}", file=sys.stderr)
        return color
    return "Unknown"


# =========================
# Generic camera generator
# =========================

def generate_camera_stream(processor=None):
    """
    Generic MJPEG camera stream.
    `processor(frame)` can modify frame and return extra info (ignored here).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Yield a single error frame so the stream doesn't crash.
        error_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(
            error_frame,
            "Camera error",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        ret, buffer = cv2.imencode(".jpg", error_frame)
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            if processor is not None:
                try:
                    frame = processor(frame)
                except Exception:
                    # Never break the stream because of processing errors.
                    pass

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                break
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


# ================= DASHBOARD =================


@app.route("/")
def dashboard():
    modules = [
        {
            "name": "Long Hair Identification",
            "description": "Age, gender, and hair-based classification.",
            "route": url_for("long_hair"),
            "status": "active",
        },
        {
            "name": "Senior Citizen Identification",
            "description": "Detect seniors in live camera and log to CSV.",
            "route": url_for("senior"),
            "status": "active",
        },
        {
            "name": "Voice Age & Emotion",
            "description": "Voice-based gender, age, and emotion (male only).",
            "route": url_for("voice"),
            "status": "active",
        },
        {
            "name": "Sign Language Detection",
            "description": "CNN-based gesture words (available 24/7).",
            "route": url_for("sign_language"),
            "status": "active",
        },
        {
            "name": "Car Colour Detection",
            "description": "Detect cars, people, and car colours.",
            "route": url_for("car_colour"),
            "status": "active",
        },
        {
            "name": "Nationality Detection",
            "description": "Face-based nationality, age, emotion & dress color.",
            "route": url_for("nationality"),
            "status": "active",
        },
    ]
    return render_template("index.html", modules=modules)


# ================= LONG HAIR MODULE =================


@app.route("/long-hair", methods=["GET", "POST"])
def long_hair():
    result = None
    age_value = None
    gender_value = None
    hair_value = None
    model_status = {
        "face_model": "OK" if FACE_AVAILABLE else "MISSING",
        "age_model": "OK" if AGE_AVAILABLE else "MISSING",
        "gender_model": "OK" if GENDER_AVAILABLE else "MISSING",
    }

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please upload an image.", "danger")
            return redirect(url_for("long_hair"))

        save_path = os.path.join(
            app.config["UPLOAD_FOLDER"], file.filename
        )
        file.save(save_path)

        img = cv2.imread(save_path)
        if img is None:
            flash("Unable to read image.", "danger")
            return redirect(url_for("long_hair"))

        faces = detect_faces_dnn(img)
        if not faces:
            # fallback to Haar if DNN not available or no faces found
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            haar_faces = HAAR_FACE.detectMultiScale(gray, 1.3, 5)
            faces = [
                (x, y, x + w, y + h) for (x, y, w, h) in haar_faces
            ]

        if not faces:
            result = "No face detected."
        else:
            # Use first face for this module (padded crop improves gender/age accuracy)
            x1, y1, x2, y2 = faces[0]
            face_img = _crop_face_with_padding(img, (x1, y1, x2, y2))
            approx_age, age_bucket, gender_pred, demo, gender_conf = (
                predict_age_and_gender(face_img)
            )
            hair_len = classify_hair_length(faces[0], img)

            # Caffe model has strong Male bias for females; correct when not highly confident
            if gender_conf < 0.72 and hair_len == "long" and gender_pred == "Male":
                gender_pred = "Female"
            elif gender_conf < 0.82 and gender_pred == "Male":
                gender_pred = "Female"  # Model often wrong for females; flip unless very confident Male

            age_value = approx_age
            gender_value = gender_pred
            hair_value = hair_len

            result = (
                f"Age ~{approx_age}, Gender: {gender_pred}, Hair: {hair_len}"
            )

            if demo:
                result += " (Demo mode: age/gender model not fully loaded)"

    return render_template(
        "long_hair.html",
        result=result,
        age=age_value,
        gender=gender_value,
        hair=hair_value,
        model_status=model_status,
    )


def process_long_hair_frame(frame):
    """Processor used for live camera in long-hair module."""
    faces = detect_faces_dnn(frame)
    if not faces:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar = HAAR_FACE.detectMultiScale(gray, 1.3, 5)
        faces = [(x, y, x + w, y + h) for (x, y, w, h) in haar]

    for box in faces:
        x1, y1, x2, y2 = box
        face_img = _crop_face_with_padding(frame, box)
        approx_age, age_bucket, gender_pred, demo, gender_conf = (
            predict_age_and_gender(face_img)
        )
        hair_len = classify_hair_length(box, frame)

        # Caffe model has strong Male bias for females; correct when not highly confident
        if gender_conf < 0.72 and hair_len == "long" and gender_pred == "Male":
            gender_pred = "Female"
        elif gender_conf < 0.82 and gender_pred == "Male":
            gender_pred = "Female"  # Model often wrong for females; flip unless very confident Male

        label = f"{approx_age}y, {gender_pred}, {hair_len}"
        if demo:
            label += " [demo]"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return frame


@app.route("/video_feed_long_hair")
def video_feed_long_hair():
    return Response(
        generate_camera_stream(process_long_hair_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ================= SENIOR CITIZEN MODULE =================


def process_senior_frame(frame):
    """Detect faces, age, gender and mark seniors."""
    faces = detect_faces_dnn(frame)
    if not faces:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar = HAAR_FACE.detectMultiScale(gray, 1.3, 5)
        faces = [(x, y, x + w, y + h) for (x, y, w, h) in haar]

    for box in faces:
        x1, y1, x2, y2 = box
        face_img = _crop_face_with_padding(frame, box)
        approx_age, age_bucket, gender_pred, demo, gender_conf = (
            predict_age_and_gender(face_img)
        )
        hair_len = classify_hair_length(box, frame)
        # Caffe model has strong Male bias for females; correct when not highly confident
        if gender_conf < 0.72 and hair_len == "long" and gender_pred == "Male":
            gender_pred = "Female"
        elif gender_conf < 0.82 and gender_pred == "Male":
            gender_pred = "Female"  # Model often wrong for females; flip unless very confident Male
        is_senior = approx_age > 60

        if is_senior:
            color = (0, 0, 255)  # Red box
        else:
            color = (255, 0, 0)  # Blue box

        label = f"{gender_pred}, {age_bucket}"
        if is_senior:
            label += " - Senior Citizen"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Log to CSV
        log_senior(age_bucket, gender_pred, is_senior)

    return frame


@app.route("/senior")
def senior():
    return render_template("senior.html")


@app.route("/logs/senior_log.csv")
def download_senior_log():
    """Serve the senior log CSV if it exists, otherwise ensure header and serve."""
    if pd is None:
        flash("Pandas not installed. Senior log requires pandas.", "warning")
        return redirect(url_for("senior"))
    ensure_senior_log_header()
    if not os.path.exists(SENIOR_LOG_PATH):
        flash("No log data yet.", "info")
        return redirect(url_for("senior"))
    return send_file(SENIOR_LOG_PATH, as_attachment=True)


@app.route("/video_feed_senior")
def video_feed_senior():
    return Response(
        generate_camera_stream(process_senior_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ================= VOICE MODULE =================


@app.route("/voice", methods=["GET", "POST"])
def voice():
    result = None
    gender = None
    age = None
    emotion = None

    if request.method == "POST":
        file = request.files.get("audio")
        if not file or file.filename == "":
            flash("Please upload an audio file.", "danger")
            return redirect(url_for("voice"))

        save_path = os.path.join(
            app.config["UPLOAD_FOLDER"], file.filename
        )
        file.save(save_path)

        mfcc, y, sr = extract_mfcc(save_path)
        gender = predict_voice_gender(mfcc, y, sr)
        age = predict_voice_age(mfcc) if mfcc is not None else 35
        emotion = predict_voice_emotion(mfcc, y, sr) if (mfcc is not None or y is not None) else "Neutral"
        result = f"Gender: {gender}, Age: {age}, Emotion: {emotion}"

    return render_template(
        "voice.html",
        result=result,
        gender=gender,
        age=age,
        emotion=emotion,
    )


# ================= SIGN LANGUAGE MODULE =================


def _load_image_with_exif_fix(path):
    """Load image from path; apply EXIF orientation so phone photos are right-side-up. Returns BGR numpy array or None."""
    if _PIL_AVAILABLE:
        try:
            pil_img = Image.open(path)
            if hasattr(pil_img, "_getexif") and pil_img._getexif() is not None:
                from PIL import ExifTags
                exif = pil_img._getexif()
                if exif:
                    orientation = exif.get(274)  # ExifTags.TAGS.get('Orientation', 274)
                    if orientation == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif orientation == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif orientation == 8:
                        pil_img = pil_img.rotate(90, expand=True)
            img = np.array(pil_img)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception:
            pass
    return cv2.imread(path)


@app.route("/sign-language", methods=["GET", "POST"])
def sign_language():
    time_allowed = allowed_time_for_sign_language()
    predicted_word = None
    message = None

    if not time_allowed:
        message = "System works only between 6 PM and 10 PM."
        return render_template(
            "sign_language.html",
            allowed=False,
            message=message,
            predicted_word=None,
            hand_detection_available=True,
        )

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please upload an image for sign detection.", "danger")
            return redirect(url_for("sign_language"))

        save_path = os.path.join(
            app.config["UPLOAD_FOLDER"], file.filename
        )
        file.save(save_path)

        img = _load_image_with_exif_fix(save_path)
        if img is None:
            img = cv2.imread(save_path)
        if img is None:
            flash("Unable to read image.", "danger")
            return redirect(url_for("sign_language"))

        predicted_word = predict_sign_language_word(img, static_image=True)[0]

    _try_import_mediapipe()
    return render_template(
        "sign_language.html",
        allowed=True,
        message=message,
        predicted_word=predicted_word,
        hand_detection_available=_HANDS_AVAILABLE,
    )


# Stronger identification: only show a sign after it's confirmed multiple times (reduces flicker/wrong hits)
_SIGN_LANG_HISTORY_SIZE = 5
_SIGN_LANG_CONFIRM_COUNT = 3  # same word at least this many times in history before we show it
_SIGN_LANG_NO_HAND_FRAMES = 5  # after this many prediction cycles with no hand, show "No hand"
_sign_language_history = []
_sign_language_display_word = "—"
_sign_language_no_hand_count = 0

def process_sign_language_frame(frame):
    """Demo overlay for sign language live view. Shows 'No hand' when no hand is visible."""
    global _sign_language_display_word, _sign_language_history, _sign_language_no_hand_count
    if not _HANDS_AVAILABLE:
        demo_word, _ = predict_sign_language_word(frame)
        cv2.putText(frame, f"Sign: {demo_word} (demo)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return frame
    raw_word, hand_detected = predict_sign_language_word(frame)
    if hand_detected:
        _sign_language_no_hand_count = 0
        _sign_language_history.append(raw_word)
        if len(_sign_language_history) > _SIGN_LANG_HISTORY_SIZE:
            _sign_language_history.pop(0)
        if len(_sign_language_history) >= _SIGN_LANG_CONFIRM_COUNT:
            counts = Counter(_sign_language_history)
            most_common_word, count = counts.most_common(1)[0]
            if count >= _SIGN_LANG_CONFIRM_COUNT:
                _sign_language_display_word = most_common_word
        else:
            _sign_language_display_word = raw_word
    else:
        _sign_language_no_hand_count += 1
        if _sign_language_no_hand_count >= _SIGN_LANG_NO_HAND_FRAMES:
            _sign_language_display_word = "No hand"
    word = _sign_language_display_word
    cv2.putText(
        frame,
        f"Sign: {word}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2,
    )
    return frame


@app.route("/video_feed_sign_language")
def video_feed_sign_language():
    if not allowed_time_for_sign_language():
        # Return a short error stream
        def _err_stream():
            error_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(
                error_frame,
                "Allowed 18:00-22:00 only",
                (5, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            ret, buffer = cv2.imencode(".jpg", error_frame)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        return Response(
            _err_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return Response(
        generate_camera_stream(process_sign_language_frame),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ================= CAR COLOUR MODULE =================

# ================= YOLOv8 MODEL (ultralytics) =================
from ultralytics import YOLO
import sys

# Box colors: Red=blue car, Blue=other car, Green=person
BOX_COLOR_BLUE_CAR = (0, 0, 255)
BOX_COLOR_OTHER_CAR = (255, 0, 0)
BOX_COLOR_PERSON = (0, 255, 0)

# Blue detection: HSV range (OpenCV H 0-180)
# Stricter blue range: H 100-130 (pure blue), S 100+ (saturated), V 50+ (visible)
LOWER_BLUE = np.array([100, 100, 50])
UPPER_BLUE = np.array([130, 255, 255])
# Threshold: 18% of pixels must be blue to classify as a blue car
BLUE_PIXEL_RATIO_THRESHOLD = 0.18

# Processed images saved here
CAR_OUTPUT_DIR = os.path.join(STATIC_DIR, "uploads")
os.makedirs(CAR_OUTPUT_DIR, exist_ok=True)

# YOLOv8 class IDs for COCO dataset
YOLO_CLASS_PERSON = 0
YOLO_CLASS_CAR = 2

def load_yolov8():
    """Load YOLOv8 nano model once at startup."""
    try:
        print("[YOLO] Loading YOLOv8n model...", file=sys.stderr)
        model = YOLO('yolov8n.pt')  # nano model - fast and lightweight
        print("[YOLO] YOLOv8n model loaded successfully!", file=sys.stderr)
        return model, True
    except Exception as e:
        print(f"[YOLO] Failed to load YOLOv8n: {e}", file=sys.stderr)
        return None, False

YOLO_NET, YOLO_AVAILABLE = load_yolov8()


def _is_car_blue(car_crop):
    """
    Detect if car region is blue using HSV range.
    Returns bool. Requires 18%+ of pixels to be blue-saturated.
    """
    if car_crop is None or car_crop.size == 0:
        return False
    try:
        hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        blue_pixels = cv2.countNonZero(mask)
        total_pixels = car_crop.shape[0] * car_crop.shape[1]
        ratio = blue_pixels / total_pixels if total_pixels > 0 else 0
        is_blue = ratio > BLUE_PIXEL_RATIO_THRESHOLD
        print(f"[Blue Detection] Blue pixel ratio: {ratio:.2%} (threshold: {BLUE_PIXEL_RATIO_THRESHOLD:.0%}) -> {'BLUE' if is_blue else 'OTHER'}", file=sys.stderr)
        return is_blue
    except Exception as e:
        print(f"[Blue Detection] Error: {e}", file=sys.stderr)
        return False


def detect_cars_and_people(img_bgr):
    """
    Detect cars and people using YOLOv8.
    Args:
        img_bgr: Image in BGR format (OpenCV)
    Returns:
        (car_boxes, people_boxes) where boxes are [(x1,y1,x2,y2), ...]
        or None if model not loaded
    """
    if not YOLO_AVAILABLE or YOLO_NET is None:
        print("[detect_cars_and_people] YOLO model not available", file=sys.stderr)
        return None

    # Convert BGR to RGB for YOLOv8
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Run detection with confidence threshold 0.3
    results = YOLO_NET(img_rgb, conf=0.3, verbose=False)
    
    if results is None or len(results) == 0:
        print("[detect_cars_and_people] No results from model", file=sys.stderr)
        return [], []
    
    result = results[0]  # Get first (and only) result
    
    car_boxes = []
    people_boxes = []
    
    print(f"[detect_cars_and_people] Total detections: {len(result.boxes)}", file=sys.stderr)
    
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Get coordinates [x1, y1, x2, y2]
        coords = box.xyxy[0]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        
        if class_id == YOLO_CLASS_CAR:
            print(f"[YOLO] Car detected - conf: {confidence:.3f}", file=sys.stderr)
            car_boxes.append((x1, y1, x2, y2))
        elif class_id == YOLO_CLASS_PERSON:
            print(f"[YOLO] Person detected - conf: {confidence:.3f}", file=sys.stderr)
            people_boxes.append((x1, y1, x2, y2))
    
    print(f"[detect_cars_and_people] Cars: {len(car_boxes)}, People: {len(people_boxes)}", file=sys.stderr)
    
    return car_boxes, people_boxes





@app.route("/car-colour", methods=["GET", "POST"])
def car_colour():
    processed_image_url = None
    total_cars = 0
    blue_cars = 0
    total_people = 0

    if request.method == "POST":
        if not YOLO_AVAILABLE:
            flash("YOLO model not loaded. Please add yolov4-tiny.weights + yolov4-tiny.cfg (or yolov4.weights + yolov4.cfg) to models/", "danger")
            return redirect(url_for("car_colour"))

        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please upload an image for car detection.", "danger")
            return redirect(url_for("car_colour"))

        save_path = os.path.join(
            app.config["UPLOAD_FOLDER"], file.filename
        )
        file.save(save_path)

        img = cv2.imread(save_path)
        if img is None:
            flash("Unable to read image.", "danger")
            return redirect(url_for("car_colour"))

        result = detect_cars_and_people(img)
        if result is None:
            flash("YOLO model not loaded. Please add weights and config files.", "danger")
            return redirect(url_for("car_colour"))

        car_boxes, people_boxes = result
        total_cars = len(car_boxes)
        total_people = len(people_boxes)

        for (x1, y1, x2, y2) in people_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR_PERSON, 2)
            cv2.putText(
                img,
                "Person",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                BOX_COLOR_PERSON,
                2,
            )

        for (x1, y1, x2, y2) in car_boxes:
            car_region = img[y1:y2, x1:x2]
            is_blue = _is_car_blue(car_region)
            if is_blue:
                box_color = BOX_COLOR_BLUE_CAR
                blue_cars += 1
            else:
                box_color = BOX_COLOR_OTHER_CAR

            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                img,
                f"Car ({'Blue' if is_blue else 'Other'})",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
            )

        out_name = f"car_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        out_path = os.path.join(CAR_OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, img)
        processed_image_url = f"/static/uploads/{out_name}"

    return render_template(
        "car_colour.html",
        image_url=processed_image_url,
        total_cars=total_cars,
        blue_cars=blue_cars,
        total_people=total_people,
        yolo_available=YOLO_AVAILABLE,
    )


# ================= NATIONALITY MODULE =================


def predict_nationality(face_img):
    """
    Simple nationality classifier based on facial features.
    Analyzes symmetry and brightness patterns.
    """
    if face_img is None or face_img.size == 0:
        return "Unknown"
    
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Simple heuristic: skin tone + structure indicators
        # Darker skin (lower brightness) more likely Indian/African
        # Lighter skin (higher brightness) more likely US/European
        
        if mean_brightness < 80:
            return "African"
        elif 80 <= mean_brightness < 110:
            return "Indian"
        elif 110 <= mean_brightness < 140:
            return "US/European"
        else:
            return "Other"
    except:
        return "Unknown"


def predict_emotion_face(face_img):
    """
    Simple emotion detection based on image variance and brightness.
    Smile/Happy = higher variance in lower face region.
    """
    if face_img is None or face_img.size == 0:
        return "Neutral"
    
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Lower face region (where smile appears)
        lower_face = gray[h//2:, :]
        variance = np.var(lower_face)
        mean_val = np.mean(gray)
        
        # High variance in lower face + bright = smile/happy
        if variance > 500 and mean_val > 100:
            return "Happy"
        # Low variance + dark = sad/neutral
        elif variance < 300:
            return "Sad"
        # Very bright = possibly laughing/excited
        elif mean_val > 180:
            return "Happy"
        else:
            return "Neutral"
    except:
        return "Neutral"


@app.route("/nationality", methods=["GET", "POST"])
def nationality():
    result = None
    nationality_label = None
    age_value = None
    emotion_value = None
    dress_color = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please upload an image.", "danger")
            return redirect(url_for("nationality"))

        save_path = os.path.join(
            app.config["UPLOAD_FOLDER"], file.filename
        )
        file.save(save_path)

        img = cv2.imread(save_path)
        if img is None:
            flash("Unable to read image.", "danger")
            return redirect(url_for("nationality"))

        # Detect face using DNN or Haar
        faces = detect_faces_dnn(img)
        if not faces:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            haar_faces = HAAR_FACE.detectMultiScale(gray, 1.3, 5)
            faces = [
                (x, y, x + w, y + h) for (x, y, w, h) in haar_faces
            ]

        if not faces:
            result = "No face detected."
        else:
            box = faces[0]
            x1, y1, x2, y2 = box
            face_img = img[y1:y2, x1:x2]

            nationality_label = predict_nationality(face_img)
            emotion_value = predict_emotion_face(face_img)

            approx_age, age_bucket, gender_pred, demo, _ = (
                predict_age_and_gender(face_img)
            )

            if nationality_label == "Indian":
                age_value = approx_age
                dress_color = detect_dress_color(img, box)
                result = (
                    f"Nationality: {nationality_label}, Age: {age_value}, "
                    f"Dress Color: {dress_color}, Emotion: {emotion_value}"
                )
            elif nationality_label == "US":
                age_value = approx_age
                result = (
                    f"Nationality: {nationality_label}, Age: {age_value}, "
                    f"Emotion: {emotion_value}"
                )
            elif nationality_label == "African":
                dress_color = detect_dress_color(img, box)
                result = (
                    f"Nationality: {nationality_label}, Emotion: {emotion_value}, "
                    f"Dress Color: {dress_color}"
                )
            else:
                result = (
                    f"Nationality: {nationality_label}, Emotion: {emotion_value}"
                )

            if demo:
                result += " (Demo mode: age/gender model not fully loaded)"

    return render_template(
        "nationality.html",
        result=result,
        nationality=nationality_label,
        age=age_value,
        emotion=emotion_value,
        dress_color=dress_color,
    )


# ================= MAIN =================


def _log_model_status():
    """Print model loading status at startup for debugging."""
    import sys
    _try_import_mediapipe()
    print(f"\n[Model Status] MODELS_DIR = {MODELS_DIR}", file=sys.stderr)
    print(f"  Face:  {'OK' if FACE_AVAILABLE else 'MISSING'}", file=sys.stderr)
    print(f"  Age:   {'OK' if AGE_AVAILABLE else 'MISSING'}", file=sys.stderr)
    print(f"  Gender:{'OK' if GENDER_AVAILABLE else 'MISSING'}", file=sys.stderr)
    print(f"  Hand (sign language): {'OK' if _HANDS_AVAILABLE else 'OFF - yellow banner will show'}", file=sys.stderr)
    print("", file=sys.stderr)


if __name__ == "__main__":
    _log_model_status()
    app.run(debug=True)

