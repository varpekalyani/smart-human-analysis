import cv2
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def generate_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, "Face Detected",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,0),
                            2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
