import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path


TRAINER_DIR = Path("trainer")
ATTENDANCE_DIR = Path("attendance")
RESULTS_DIR = Path("results")
MODEL_PATH = TRAINER_DIR / "face_model.yml"
LABELS_PATH = TRAINER_DIR / "labels.txt"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


ATTENDANCE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    print("Trained model or labels file not found.")
    print("Please run 02_train_model.py first.")
    raise SystemExit(1)

labels = {}

with LABELS_PATH.open("r") as file:
    for line in file:
        user_id, user_name = line.strip().split(",", 1)
        labels[int(user_id)] = user_name

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
camera = cv2.VideoCapture(0)

attendance_file = ATTENDANCE_DIR / f"attendance_{datetime.now().date()}.csv"
marked_users = set()

print("Recognition started. Press q to stop.")

while True:
    success, frame = camera.read()

    if not success:
        print("Camera frame not received.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face_image = gray_frame[y:y + h, x:x + w]
        user_id, confidence = recognizer.predict(face_image)

        if confidence < 70:
            user_name = labels.get(user_id, "Unknown")
            display_text = f"{user_name} ({round(confidence, 2)})"
            box_color = (0, 255, 0)

            if user_id not in marked_users:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_row = pd.DataFrame(
                    [[user_id, user_name, current_time]],
                    columns=["User ID", "Name", "Time"]
                )
                attendance_row.to_csv(
                    attendance_file,
                    mode="a",
                    header=not attendance_file.exists(),
                    index=False
                )
                marked_users.add(user_id)

                result_path = RESULTS_DIR / f"{user_name}_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(str(result_path), frame)
        else:
            display_text = "Unknown"
            box_color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(
            frame,
            display_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2
        )

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Recognition stopped.")
print(f"Attendance saved at: {attendance_file}")
