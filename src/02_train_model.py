import cv2
import numpy as np
from pathlib import Path


DATASET_DIR = Path("dataset")
TRAINER_DIR = Path("trainer")
MODEL_PATH = TRAINER_DIR / "face_model.yml"
LABELS_PATH = TRAINER_DIR / "labels.txt"


TRAINER_DIR.mkdir(exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}

for user_folder in DATASET_DIR.iterdir():
    if not user_folder.is_dir():
        continue

    folder_parts = user_folder.name.split("_", 1)
    user_id = int(folder_parts[0])
    user_name = folder_parts[1] if len(folder_parts) > 1 else f"User_{user_id}"
    label_names[user_id] = user_name

    for image_path in user_folder.glob("*.jpg"):
        face_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if face_image is None:
            continue

        faces.append(face_image)
        labels.append(user_id)

if len(faces) == 0:
    print("No training images found. Please run 01_capture_images.py first.")
    raise SystemExit(1)

recognizer.train(faces, np.array(labels))
recognizer.write(str(MODEL_PATH))

with LABELS_PATH.open("w") as file:
    for user_id, user_name in label_names.items():
        file.write(f"{user_id},{user_name}\n")

print("Model training completed.")
print(f"Total face images used: {len(faces)}")
print(f"Model saved at: {MODEL_PATH}")
print(f"Labels saved at: {LABELS_PATH}")
