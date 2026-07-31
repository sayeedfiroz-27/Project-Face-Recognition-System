# Face Recognition System Project

This project teaches a complete beginner-friendly Face Recognition System using Python, OpenCV, and basic Machine Learning.

## Features

- Face Detection using OpenCV
- Capture Images from Webcam
- Face Recognition using Machine Learning
- Train Face Recognition Model
- Identify Registered Users
- Real-time Face Detection
- Attendance Marking
- Save Recognition Results

## Install

Install all required modules:

```bash
python3 -m pip install -r requirements.txt
```

Or install modules one by one:

```bash
python3 -m pip install opencv-contrib-python
python3 -m pip install numpy
python3 -m pip install pandas
```

Check installation:

```bash
python3 -c "import cv2, numpy, pandas; print('All modules installed successfully')"
```

Important: this project uses `opencv-contrib-python` because LBPH face recognition needs the `cv2.face` module.

## Run Step by Step

Capture images:

```bash
python3 src/01_capture_images.py
```

Train model:

```bash
python3 src/02_train_model.py
```

Run real-time recognition:

```bash
python3 src/03_recognize_and_attendance.py
```

## Folders

- `dataset/` stores captured face images.
- `trainer/` stores trained model and user mapping.
- `attendance/` stores attendance CSV files.
- `results/` stores recognition screenshots.

## Important Image and Data Path Notes

Real face images are not included in this GitHub repository for privacy. When you run `python3 src/01_capture_images.py`, the project automatically creates user image folders like `dataset/1_Rahul/` and saves captured face images as `1.jpg`, `2.jpg`, `3.jpg`, and so on.

Generated project files:

```text
dataset/1_Rahul/1.jpg
trainer/face_model.yml
trainer/labels.txt
attendance/attendance_YYYY-MM-DD.csv
results/Rahul_HHMMSS.jpg
```

For the detailed classroom notes, open `index.html` or read `face_recognition_system_notes.md`.
