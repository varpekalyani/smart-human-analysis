import cv2
import pandas as pd
import datetime

def detect_seniors():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Load a face detector (Haar Cascade for simplicity)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Prepare CSV logging
    log_file = "data/visitors.csv"
    df = pd.DataFrame(columns=["Age", "Gender", "Time"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Placeholder logic: assume random age/gender for demo
            age = 65   # Replace with ML model prediction
            gender = "Male"

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Mark senior citizen if age > 60
            label = f"{gender}, Age {age}"
            if age > 60:
                label += " (Senior Citizen)"

            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Log to CSV
            df = pd.concat([df, pd.DataFrame([[age, gender, datetime.datetime.now()]], 
                                             columns=["Age", "Gender", "Time"])], ignore_index=True)

        cv2.imshow("Senior Citizen Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save log
    df.to_csv(log_file, index=False)

    return "Detection finished. Results saved to visitors.csv"