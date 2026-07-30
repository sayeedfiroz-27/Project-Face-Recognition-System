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

```bash
python3 -m pip install -r requirements.txt
```

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
