# Day 13 - Mini Project 4

# Project: Face Recognition System

## Topics Covered

Face Detection using OpenCV, Capture Images from Webcam, Face Recognition using Machine Learning, Train Face Recognition Model, Identify Registered Users, Real-time Face Detection, Basic Attendance Marking, and Save Recognition Results.

Is project ka goal hai students ko ek complete real-world Computer Vision project step by step build karwana. Aaj hum sirf theory nahi padhenge. Hum webcam se images capture karenge, face detector use karenge, captured faces se model train karenge, phir camera ke saamne registered user ko identify karenge, attendance CSV me save karenge, aur recognition result image ke form me store karenge.

Teacher speaking flow: "Students, face recognition system ek real-world project hai. Isme pehle face detect hota hai, phir face images collect hoti hain, phir model train hota hai, phir live camera me user identify hota hai. Aaj ka focus sirf code run karna nahi hai. Hum har step ka reason samjhenge: kyu face grayscale hota hai, kyu multiple images chahiye, kyu model train hota hai, confidence score kya hota hai, aur attendance file kaise save hoti hai."

Important privacy note: Face recognition sensitive technology hai. Is project ko sirf learning, classroom demo, aur consent-based practice ke liye use karna chahiye. Kisi ka face data bina permission capture ya use nahi karna chahiye. Real company ya school me face recognition use karne se pehle privacy policy, consent, and data security zaroor follow karni hoti hai.

---

# 1. Project Purpose and Real-World Story

Face Recognition System ka purpose hai camera ke through kisi registered person ko identify karna. Example ke liye classroom attendance system socho. Pehle har student ke face images register kiye jayenge. Phir model ko train kiya jayega. Jab student camera ke saamne aayega, system face detect karega, trained model se compare karega, name show karega, attendance mark karega, aur result save karega.

Is project me three main files hain. `src/01_capture_images.py` registered user ke face images capture karta hai. `src/02_train_model.py` captured images se ML model train karta hai. `src/03_recognize_and_attendance.py` trained model use karke live camera me user ko identify karta hai aur attendance save karta hai.

## Kab Use Hota Hai

Face recognition attendance, office entry system, gym membership verification, lab access control, classroom demo, smart camera apps, and identity-based automation me use hota hai. Lekin hamesha consent and privacy important hoti hai. Learning project me hum local webcam and local files use kar rahe hain, taaki students concept samajh sakein without cloud upload.

---

# 2. Module Download and Setup

Project start karne se pehle modules install karna zaroori hai. Module ka matlab ready-made Python package hota hai jo extra features provide karta hai. Python ke basic installation me webcam access, face detection, face recognition model, CSV attendance report, and image processing ke tools built-in nahi hote. Isliye hum required packages install karte hain.

Is project me `opencv-contrib-python`, `numpy`, and `pandas` use honge. `opencv-contrib-python` sabse important hai because isme `cv2.face.LBPHFaceRecognizer_create()` available hota hai. Agar sirf `opencv-python` install hai, to kai baar `cv2.face` module missing aata hai. `numpy` labels ko numerical array me convert karne ke liye use hota hai. `pandas` attendance CSV create and append karne ke liye use hota hai.

## Install All Modules Together

```bash
python3 -m pip install -r requirements.txt
```

## Install Modules One by One

```bash
python3 -m pip install opencv-contrib-python
python3 -m pip install numpy
python3 -m pip install pandas
```

## Check Installation

```bash
python3 -c "import cv2, numpy, pandas; print('All modules installed successfully')"
```

## Detailed Command Explanation

| Part | Code | Explanation |
|---|---|---|
| 1 | `python3` | Ye Python 3 interpreter use karta hai. Mac/Linux me mostly `python3` command hoti hai. Windows me agar `python` Python 3 open karta hai to `python -m pip ...` bhi use ho sakta hai. |
| 2 | `-m pip` | Ye Python ko bolta hai ki pip module run karo. `pip` packages install karne ka tool hai. `python3 -m pip` same Python environment me packages install karne me help karta hai. |
| 3 | `install` | Ye pip ko instruction deta hai ki packages install karne hain. |
| 4 | `-r requirements.txt` | Ye pip ko bolta hai ki package list `requirements.txt` file se read karo. Isse ek command me required modules install ho jate hain. |
| 5 | `opencv-contrib-python` | Ye OpenCV ka contrib package hai. Webcam, image processing, Haar Cascade, and LBPH face recognizer ke liye required hai. |
| 6 | `numpy` | Ye numerical arrays ke liye use hota hai. Training labels ko model ke format me convert karne ke liye NumPy use hota hai. |
| 7 | `pandas` | Ye table data and CSV handling ke liye use hota hai. Attendance file create karne ke liye Pandas use hoga. |

Teacher speaking flow: "Students, correct code likhne ke baad bhi error aa sakta hai agar module install nahi hai. Isliye pehle environment ready karo. Face recognition ke liye especially `opencv-contrib-python` install hona zaroori hai, kyunki LBPH recognizer contrib package me hota hai."

---

# 3. Important Project Folders

Face Recognition project me folder structure bahut important hai. Code files ke saath-saath images, trained model, attendance CSV, and recognition screenshots bhi generate honge.

`dataset/` folder captured face images store karega. Har user ke liye separate folder banega, jaise `dataset/1_Rahul/`. Is folder me `1.jpg`, `2.jpg`, `3.jpg` jaise cropped grayscale face images save hongi.

`trainer/` folder trained model store karega. Training ke baad `trainer/face_model.yml` model file save hogi. `trainer/labels.txt` file user ID and user name mapping store karegi, jaise `1,Rahul`.

`attendance/` folder daily attendance CSV files store karega. Recognition ke time file name `attendance_YYYY-MM-DD.csv` jaisa banega.

`results/` folder successful recognition screenshots store karega. Jab user recognize hota hai, current frame image ke form me save ho sakta hai, jaise `results/Rahul_143025.jpg`.

## Kab Use Hota Hai

Folders tab use hote hain jab project data ko organized way me save karna ho. Dataset images alag, trained model alag, attendance report alag, and results alag rakhne se project professional and manageable banta hai. Agar sab files same folder me daal denge to debugging and explanation difficult ho jayega.

---

# 4. Face Detection using OpenCV

Face Detection ka matlab image ya video frame me human face ka location find karna. Face detection sirf batata hai face kaha hai. Ye nahi batata face kis person ka hai. Agar camera me Rahul ka face hai, face detection rectangle draw karega, lekin Rahul ka naam nahi batayega. Naam batana face recognition ka kaam hai.

OpenCV ek Computer Vision library hai. Computer Vision ka matlab computer ko images and videos samjhana. OpenCV se hum webcam open kar sakte hain, frames read kar sakte hain, grayscale conversion kar sakte hain, face detect kar sakte hain, rectangle draw kar sakte hain, aur live video show kar sakte hain.

Is project me Haar Cascade detector use hoga. Haar Cascade ek pre-trained classifier hai jo face-like light-dark patterns detect karta hai. Pre-trained ka matlab hume detector ko train nahi karna; OpenCV ke andar ready XML file already available hoti hai.

## Kab Use Hota Hai

Face detection tab use hota hai jab hume image/video me face ka area find karna ho. Attendance system me pehle face detect hota hai, phir detected face crop hota hai, phir recognition model us cropped face ko identify karta hai. Face detection ke bina recognition model ko pata hi nahi chalega ki full image me face kaha hai.

## Practice Code 1 - Basic Face Detection

```python
import cv2

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
camera = cv2.VideoCapture(0)

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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
```

## Output

```text
Webcam window open hogi.
Face detect hone par green rectangle show hoga.
q press karne par program close hoga.
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `import cv2` | OpenCV library import hoti hai. Webcam access, image conversion, Haar Cascade face detection, rectangle drawing, and window display ke liye `cv2` required hai. |
| 3 | `CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"` | Ye Haar Cascade XML detector file ka path banata hai. `cv2.data.haarcascades` OpenCV ke built-in cascade folder ka path deta hai. File name frontal face detection ke liye pre-trained model hai. |
| 5 | `face_detector = cv2.CascadeClassifier(CASCADE_PATH)` | Ye cascade file ko load karke face detector object banata hai. Ye object frame ke andar face coordinates find karega. |
| 6 | `camera = cv2.VideoCapture(0)` | Ye default webcam open karta hai. `0` first camera ko represent karta hai. Agar external camera ho to `1` try kar sakte hain. |
| 8 | `while True:` | Ye infinite loop start karta hai. Live video many frames ka sequence hota hai, isliye frames continuously read karne padte hain. |
| 9 | `success, frame = camera.read()` | Webcam se current frame read hota hai. `success` batata hai frame mila ya nahi, aur `frame` actual image hoti hai. |
| 11 | `if not success:` | Ye check karta hai ki camera frame receive nahi hua. Agar frame fail hua to processing continue karna safe nahi. |
| 12 | `print("Camera frame not received.")` | User ko readable message milta hai ki frame nahi mila. |
| 13 | `break` | Loop stop karta hai because frame ke bina detection possible nahi. |
| 15 | `gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` | Color frame ko grayscale me convert karta hai. Haar Cascade brightness patterns par work karta hai, isliye grayscale faster and practical hota hai. |
| 17 | `faces = face_detector.detectMultiScale(` | Face detection function start hota hai. Ye image me one or more faces find karta hai. |
| 18 | `gray_frame,` | Detector ko grayscale image input milti hai. |
| 19 | `scaleFactor=1.2,` | Ye different face sizes handle karne ke liye image scale adjust karta hai. Face camera ke paas ho to bada, door ho to chhota dikhta hai. |
| 20 | `minNeighbors=5,` | Ye false detections reduce karta hai. Higher value stricter detection banati hai. |
| 21 | `minSize=(80, 80)` | Ye minimum face size set karta hai. Chhote noisy patterns ignore ho jate hain. |
| 22 | `)` | Detection function close hota hai aur `faces` me rectangles store ho jate hain. |
| 24 | `for (x, y, w, h) in faces:` | Har detected face ke coordinates par loop chalta hai. `x,y` starting point hain, `w,h` width and height hain. |
| 25 | `cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)` | Detected face ke around green rectangle draw hota hai. OpenCV BGR color use karta hai, so `(0,255,0)` green hai. |
| 27 | `cv2.imshow("Face Detection", frame)` | Processed frame window me show hota hai. |
| 29 | `if cv2.waitKey(1) & 0xFF == ord("q"):` | Keyboard input check hota hai. Agar user `q` press kare to condition true hoti hai. |
| 30 | `break` | User ke quit command par loop stop hota hai. |
| 32 | `camera.release()` | Webcam resource free hota hai. |
| 33 | `cv2.destroyAllWindows()` | OpenCV windows close hoti hain. |

---

# 5. Capture Images from Webcam

Face recognition model ko train karne ke liye registered users ke multiple face images chahiye. Agar system Rahul ko recognize karega, to Rahul ke face ke examples dataset me hone chahiye. Model examples se learn karta hai. One image enough nahi hoti because face angle, lighting, expression, distance, and background change ho sakte hain.

Is project me capture script user se numeric ID and name lega. Phir user ke naam ka folder create karega. Webcam se face detect hoga, face crop hoga, resize hoga, and grayscale image folder me save hogi.

## Kab Use Hota Hai

Capture step registration phase me use hota hai. Jab new student/employee system me add hota hai, pehle uske face images collect karte hain. Ye images training data ban jati hain. Agar images clear and varied hongi to recognition better hoga.

## Practical Code 2 - Capture Registered User Images

```python
import cv2
from pathlib import Path

DATASET_DIR = Path("dataset")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (200, 200)

DATASET_DIR.mkdir(exist_ok=True)

user_id = input("Enter numeric user ID: ").strip()
user_name = input("Enter user name: ").strip().replace(" ", "_")

user_folder = DATASET_DIR / f"{user_id}_{user_name}"
user_folder.mkdir(exist_ok=True)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
camera = cv2.VideoCapture(0)

if face_detector.empty():
    print("Face detector could not be loaded. Please check OpenCV installation.")
    raise SystemExit(1)

if not camera.isOpened():
    print("Camera could not be opened. Please check webcam permission or camera index.")
    raise SystemExit(1)

image_count = 0
max_images = 50

print("Camera started. Look at the camera.")
print("Press q to stop early.")

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
        image_count += 1

        face_image = gray_frame[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, FACE_SIZE)
        image_path = user_folder / f"{image_count}.jpg"
        cv2.imwrite(str(image_path), face_image)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Images: {image_count}/{max_images}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Capture Face Images", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if image_count >= max_images:
        break

camera.release()
cv2.destroyAllWindows()

print(f"Image capture completed for {user_name}.")
print(f"Total images saved: {image_count}")
```

## Output

```text
Enter numeric user ID: 1
Enter user name: Rahul
Camera window open hogi.
Face detect hone par cropped images dataset/1_Rahul/ me save hongi.
50 images complete hone par program close ho jayega.
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `import cv2` | OpenCV import hota hai. Webcam, face detection, resize, image save, rectangle, text, and display sab OpenCV se hoga. |
| 2 | `from pathlib import Path` | `Path` folder and file paths ko clean way me handle karta hai. Isse dataset folder and user folder banana easy hota hai. |
| 4 | `DATASET_DIR = Path("dataset")` | Ye dataset folder ka path define karta hai jahan captured face images save hongi. |
| 5 | `CASCADE_PATH = ...` | Haar Cascade face detector XML ka path define hota hai. |
| 6 | `FACE_SIZE = (200, 200)` | Har cropped face ko same 200x200 size me resize karne ke liye fixed size set hota hai. Same size training ke liye important hai. |
| 8 | `DATASET_DIR.mkdir(exist_ok=True)` | Dataset folder create hota hai. Agar already exist karta hai to error nahi aata. |
| 10 | `user_id = input(...).strip()` | User se numeric ID li jati hai. Model labels numeric hote hain, isliye ID important hai. `.strip()` extra spaces remove karta hai. |
| 11 | `user_name = input(...).strip().replace(" ", "_")` | User ka name liya jata hai. Spaces underscore me convert hote hain taaki folder name clean rahe. |
| 13 | `user_folder = DATASET_DIR / f"{user_id}_{user_name}"` | Current user ka folder path create hota hai, jaise `dataset/1_Rahul`. |
| 14 | `user_folder.mkdir(exist_ok=True)` | User-specific folder create hota hai. Is folder me face images save hongi. |
| 16 | `face_detector = cv2.CascadeClassifier(CASCADE_PATH)` | Haar Cascade detector load hota hai. Capture ke time face location find karne ke liye ye required hai. |
| 17 | `camera = cv2.VideoCapture(0)` | Default webcam open hota hai. |
| 19 | `if face_detector.empty():` | Check karta hai detector properly load hua ya nahi. |
| 20 | `print(...)` | Agar detector load nahi hua to error message show hota hai. |
| 21 | `raise SystemExit(1)` | Program safely stop hota hai because detector ke bina capture useful nahi. |
| 23 | `if not camera.isOpened():` | Check karta hai webcam open hua ya nahi. |
| 24 | `print(...)` | Camera issue ka readable message print hota hai. |
| 25 | `raise SystemExit(1)` | Camera unavailable ho to program stop hota hai. |
| 27 | `image_count = 0` | Saved image counter zero se start hota hai. |
| 28 | `max_images = 50` | Ek user ke liye maximum 50 images capture karne ka target set hota hai. |
| 30 | `print("Camera started. Look at the camera.")` | User ko instruction milta hai ki camera ready hai. |
| 31 | `print("Press q to stop early.")` | User ko early stop control bataya jata hai. |
| 33 | `while True:` | Continuous webcam loop start hota hai. |
| 34 | `success, frame = camera.read()` | Camera se current frame read hota hai. |
| 36 | `if not success:` | Frame fail check hota hai. |
| 37 | `print("Camera frame not received.")` | Frame issue ka message print hota hai. |
| 38 | `break` | Loop stop hota hai. |
| 40 | `gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` | Frame grayscale me convert hota hai because detector and saved training images grayscale me use honge. |
| 41 | `faces = face_detector.detectMultiScale(` | Current frame me faces detect karne ka function start hota hai. |
| 42 | `gray_frame,` | Detection ke liye grayscale image input milti hai. |
| 43 | `scaleFactor=1.2,` | Different face sizes handle hote hain. |
| 44 | `minNeighbors=5,` | False detection reduce hoti hai. |
| 45 | `minSize=(80, 80)` | Minimum face size define hota hai. |
| 46 | `)` | Detection call close hota hai. |
| 48 | `for (x, y, w, h) in faces:` | Har detected face par loop chalta hai. |
| 49 | `image_count += 1` | Image counter increase hota hai. Ye file naming ke liye use hoga. |
| 51 | `face_image = gray_frame[y:y + h, x:x + w]` | Detected face area crop hota hai. `y:y+h` row range hai aur `x:x+w` column range hai. |
| 52 | `face_image = cv2.resize(face_image, FACE_SIZE)` | Cropped face 200x200 size me resize hota hai. Same size training consistency ke liye zaroori hai. |
| 53 | `image_path = user_folder / f"{image_count}.jpg"` | Current face image ka file path create hota hai, jaise `dataset/1_Rahul/1.jpg`. |
| 54 | `cv2.imwrite(str(image_path), face_image)` | Cropped face image file ke form me save hoti hai. `str()` Path ko string me convert karta hai because OpenCV file path string expect karta hai. |
| 56 | `cv2.rectangle(...)` | Live frame me detected face ke around green rectangle draw hota hai. |
| 57 | `cv2.putText(...)` | Frame par image count text draw hota hai, jaise `Images: 10/50`. |
| 59 | `cv2.imshow("Capture Face Images", frame)` | Live camera frame screen par show hota hai. |
| 61 | `if cv2.waitKey(1) & 0xFF == ord("q"):` | Keyboard se `q` press check hota hai. |
| 62 | `break` | User `q` press kare to loop stop hota hai. |
| 64 | `if image_count >= max_images:` | Check karta hai 50 images complete hui ya nahi. |
| 65 | `break` | Target images complete hone par loop stop hota hai. |
| 67 | `camera.release()` | Webcam resource free hota hai. |
| 68 | `cv2.destroyAllWindows()` | OpenCV windows close hoti hain. |
| 70 | `print(f"Image capture completed for {user_name}.")` | Capture completion message print hota hai. |
| 71 | `print(f"Total images saved: {image_count}")` | Total saved images ka count print hota hai. |

---

# 6. Train Face Recognition Model

Training ka matlab model ko captured face images se learn karwana. Face recognition me model images ko numeric patterns ke form me samajhta hai. Is project me LBPH Face Recognizer use hoga. LBPH ka full form Local Binary Patterns Histogram hai. Simple words me, ye face image ke local texture patterns ko numbers me convert karta hai aur un patterns ko user ID ke saath learn karta hai.

Training ke time script `dataset/` folder ke andar har user folder read karega. Folder name se user ID and name niklega. Har image grayscale me read hogi, same size me resize hogi, faces list me add hogi, and label list me user ID add hoga. Finally recognizer train hoga and model `trainer/face_model.yml` me save hoga.

## Kab Use Hota Hai

Training tab use hoti hai jab new users ki images capture ho chuki hoti hain. Agar aap new student add karte ho, to images capture ke baad model ko dobara train karna chahiye. Model trained images ke basis par hi users ko identify karega.

## Practical Code 3 - Train Model

```python
import cv2
import numpy as np
from pathlib import Path

DATASET_DIR = Path("dataset")
TRAINER_DIR = Path("trainer")
MODEL_PATH = TRAINER_DIR / "face_model.yml"
LABELS_PATH = TRAINER_DIR / "labels.txt"
FACE_SIZE = (200, 200)

TRAINER_DIR.mkdir(exist_ok=True)

if not DATASET_DIR.exists():
    print("Dataset folder not found. Please run 01_capture_images.py first.")
    raise SystemExit(1)

if not hasattr(cv2, "face"):
    print("cv2.face module not found. Please install opencv-contrib-python.")
    raise SystemExit(1)

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

        face_image = cv2.resize(face_image, FACE_SIZE)
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
```

## Output

```text
Model training completed.
Total face images used: 50
Model saved at: trainer/face_model.yml
Labels saved at: trainer/labels.txt
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `import cv2` | OpenCV import hota hai. LBPH recognizer, grayscale image read, resize, and model save ke liye required hai. |
| 2 | `import numpy as np` | NumPy import hota hai. Labels list ko numeric array me convert karne ke liye use hoga. |
| 3 | `from pathlib import Path` | Folder and file paths handle karne ke liye `Path` import hota hai. |
| 5 | `DATASET_DIR = Path("dataset")` | Captured images ka main folder define hota hai. |
| 6 | `TRAINER_DIR = Path("trainer")` | Trained model output folder define hota hai. |
| 7 | `MODEL_PATH = TRAINER_DIR / "face_model.yml"` | Model save hone ka file path set hota hai. |
| 8 | `LABELS_PATH = TRAINER_DIR / "labels.txt"` | User ID and name mapping save hone ka path set hota hai. |
| 9 | `FACE_SIZE = (200, 200)` | Training images ka fixed size set hota hai. |
| 11 | `TRAINER_DIR.mkdir(exist_ok=True)` | Trainer folder create hota hai agar missing hai. |
| 13 | `if not DATASET_DIR.exists():` | Check karta hai dataset folder available hai ya nahi. |
| 14 | `print(...)` | Agar dataset missing hai to user ko pehle capture script run karne ka message milta hai. |
| 15 | `raise SystemExit(1)` | Dataset ke bina training possible nahi, isliye program stop hota hai. |
| 17 | `if not hasattr(cv2, "face"):` | Check karta hai OpenCV me `face` module available hai ya nahi. |
| 18 | `print(...)` | Agar `cv2.face` missing hai to `opencv-contrib-python` install karne ka message deta hai. |
| 19 | `raise SystemExit(1)` | Required module missing ho to program stop hota hai. |
| 21 | `recognizer = cv2.face.LBPHFaceRecognizer_create()` | LBPH face recognizer model object create hota hai. Ye captured face patterns learn karega. |
| 23 | `faces = []` | Training images store karne ke liye empty list create hoti hai. |
| 24 | `labels = []` | Har image ke matching user ID labels store karne ke liye list create hoti hai. |
| 25 | `label_names = {}` | User ID and user name mapping store karne ke liye dictionary create hoti hai. |
| 27 | `for user_folder in DATASET_DIR.iterdir():` | Dataset folder ke andar har user folder par loop chalta hai. |
| 28 | `if not user_folder.is_dir():` | Check karta hai current item folder hai ya nahi. |
| 29 | `continue` | Agar item folder nahi hai to skip hota hai. |
| 31 | `folder_parts = user_folder.name.split("_", 1)` | Folder name ko ID and name me split karta hai. Example `1_Rahul` se `1` and `Rahul` milta hai. |
| 32 | `user_id = int(folder_parts[0])` | Folder name ka first part numeric user ID me convert hota hai. |
| 33 | `user_name = ...` | Agar folder me name available hai to use karta hai, warna default `User_ID` name banata hai. |
| 34 | `label_names[user_id] = user_name` | ID and name mapping dictionary me save hoti hai. |
| 36 | `for image_path in user_folder.glob("*.jpg"):` | Current user folder ke andar all JPG images par loop chalta hai. |
| 37 | `face_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)` | Face image grayscale mode me read hoti hai. Training ke liye grayscale consistent and lightweight hota hai. |
| 39 | `if face_image is None:` | Check karta hai image file read hui ya corrupt/missing hai. |
| 40 | `continue` | Invalid image skip hoti hai. |
| 42 | `face_image = cv2.resize(face_image, FACE_SIZE)` | Image fixed 200x200 size me resize hoti hai. |
| 43 | `faces.append(face_image)` | Processed face image training list me add hoti hai. |
| 44 | `labels.append(user_id)` | Same image ka correct user ID label list me add hota hai. |
| 46 | `if len(faces) == 0:` | Check karta hai koi training image mili ya nahi. |
| 47 | `print(...)` | Agar image nahi mili to user ko capture step run karne ka message milta hai. |
| 48 | `raise SystemExit(1)` | Images ke bina training stop hoti hai. |
| 50 | `recognizer.train(faces, np.array(labels))` | Model training hoti hai. Faces input examples hain aur labels correct user IDs hain. |
| 51 | `recognizer.write(str(MODEL_PATH))` | Trained model file me save hota hai. |
| 53 | `with LABELS_PATH.open("w") as file:` | Labels mapping file write mode me open hoti hai. `with` file ko safely close karta hai. |
| 54 | `for user_id, user_name in label_names.items():` | Har registered user ID and name par loop chalta hai. |
| 55 | `file.write(f"{user_id},{user_name}\n")` | ID and name CSV-like format me labels file me save hote hain. |
| 57 | `print("Model training completed.")` | Training success message print hota hai. |
| 58 | `print(f"Total face images used: {len(faces)}")` | Total training images count print hota hai. |
| 59 | `print(f"Model saved at: {MODEL_PATH}")` | Model path print hota hai. |
| 60 | `print(f"Labels saved at: {LABELS_PATH}")` | Labels file path print hota hai. |

---

# 7. Identify Registered Users and Real-Time Face Detection

Recognition ka matlab detected face ko trained model se compare karke person identify karna. Detection face location batata hai. Recognition face identity batata hai. Is project me model face image predict karega and output me `user_id` and `confidence` dega.

Confidence score LBPH me distance jaisa hota hai. Lower confidence value usually better match indicate karti hai. Is project me threshold `70` rakha gaya hai. Agar confidence 70 se less hai to user registered maana jayega. Agar confidence high hai to user unknown show hoga.

## Kab Use Hota Hai

Recognition tab use hota hai jab model trained ho chuka ho. Real-time recognition attendance system, access system, and identity verification demo me use hota hai. Recognition se pehle detection and training dono complete hone chahiye.

---

# 8. Attendance Marking and Save Recognition Results

Attendance marking ka matlab recognized user ka ID, name, and time CSV file me save karna. Basic attendance system me ek user ko same session me baar-baar mark nahi karna chahiye. Isliye `marked_users` set use hota hai. Agar user already marked hai, system dubara same session me attendance row add nahi karega.

Save recognition results ka matlab successful recognition ka screenshot save karna. Ye proof/report ke liye useful hai. Example: `results/Rahul_143025.jpg` file show kar sakti hai ki Rahul ko 14:30:25 par recognize kiya gaya.

## Kab Use Hota Hai

Attendance marking tab use hoti hai jab recognition project ko practical classroom/office use case banana ho. Save results tab use hota hai jab hume log/proof maintain karna ho. Learning project me ye feature students ko file handling, CSV report, and real-world automation ka connection samjhata hai.

## Practical Code 4 - Recognition, Attendance, and Results

```python
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
FACE_SIZE = (200, 200)

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

if not hasattr(cv2, "face"):
    print("cv2.face module not found. Please install opencv-contrib-python.")
    raise SystemExit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
camera = cv2.VideoCapture(0)

if face_detector.empty():
    print("Face detector could not be loaded. Please check OpenCV installation.")
    raise SystemExit(1)

if not camera.isOpened():
    print("Camera could not be opened. Please check webcam permission or camera index.")
    raise SystemExit(1)

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
        face_image = cv2.resize(face_image, FACE_SIZE)
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
        cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Recognition stopped.")
print(f"Attendance saved at: {attendance_file}")
```

## Output

```text
Recognition started. Press q to stop.
Registered user detect hone par name and confidence show hoga.
Attendance CSV attendance/ folder me save hogi.
Recognition screenshot results/ folder me save hoga.
q press karne par recognition stop hoga.
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `import cv2` | OpenCV recognition model, face detection, webcam, rectangle, text, screenshot save, and display ke liye import hota hai. |
| 2 | `import pandas as pd` | Pandas attendance row ko DataFrame me convert karke CSV me save karne ke liye use hota hai. |
| 3 | `from datetime import datetime` | Current date and time nikalne ke liye use hota hai. Attendance and result screenshot file names me time chahiye. |
| 4 | `from pathlib import Path` | Folder and file paths clean way me handle karne ke liye import hota hai. |
| 6 | `TRAINER_DIR = Path("trainer")` | Trained model folder ka path define hota hai. |
| 7 | `ATTENDANCE_DIR = Path("attendance")` | Attendance CSV folder ka path define hota hai. |
| 8 | `RESULTS_DIR = Path("results")` | Recognition screenshots folder ka path define hota hai. |
| 9 | `MODEL_PATH = TRAINER_DIR / "face_model.yml"` | Trained model file ka path define hota hai. |
| 10 | `LABELS_PATH = TRAINER_DIR / "labels.txt"` | Labels mapping file ka path define hota hai. |
| 11 | `CASCADE_PATH = ...` | Haar Cascade detector file ka path define hota hai. |
| 12 | `FACE_SIZE = (200, 200)` | Recognition ke time face crop ko training ke same size me resize karne ke liye fixed size set hota hai. |
| 14 | `ATTENDANCE_DIR.mkdir(exist_ok=True)` | Attendance folder create hota hai agar missing hai. |
| 15 | `RESULTS_DIR.mkdir(exist_ok=True)` | Results folder create hota hai agar missing hai. |
| 17 | `if not MODEL_PATH.exists() or not LABELS_PATH.exists():` | Check karta hai model and labels file available hain ya nahi. Recognition ke liye dono required hain. |
| 18 | `print("Trained model or labels file not found.")` | Missing files ka error message show hota hai. |
| 19 | `print("Please run 02_train_model.py first.")` | User ko correct previous step bataya jata hai. |
| 20 | `raise SystemExit(1)` | Model missing ho to program safely stop hota hai. |
| 22 | `labels = {}` | Empty dictionary create hoti hai jisme user ID to user name mapping store hogi. |
| 24 | `with LABELS_PATH.open("r") as file:` | Labels file read mode me open hoti hai. `with` file ko automatically close karta hai. |
| 25 | `for line in file:` | Labels file ki har line read hoti hai. |
| 26 | `user_id, user_name = line.strip().split(",", 1)` | Line se ID and name separate hote hain. `.strip()` newline remove karta hai. |
| 27 | `labels[int(user_id)] = user_name` | Numeric user ID key ke saath user name dictionary me save hota hai. |
| 29 | `if not hasattr(cv2, "face"):` | Check karta hai `cv2.face` module available hai ya nahi. |
| 30 | `print(...)` | Missing contrib package ka error message print hota hai. |
| 31 | `raise SystemExit(1)` | Required module missing ho to program stop hota hai. |
| 33 | `recognizer = cv2.face.LBPHFaceRecognizer_create()` | Blank LBPH recognizer object create hota hai. |
| 34 | `recognizer.read(str(MODEL_PATH))` | Saved trained model file load hoti hai. Ab recognizer prediction ke liye ready hai. |
| 36 | `face_detector = cv2.CascadeClassifier(CASCADE_PATH)` | Face detector load hota hai. Recognition se pehle face detect karna zaroori hai. |
| 37 | `camera = cv2.VideoCapture(0)` | Webcam open hota hai. |
| 39 | `if face_detector.empty():` | Check karta hai detector load hua ya nahi. |
| 40 | `print(...)` | Detector load fail message print hota hai. |
| 41 | `raise SystemExit(1)` | Detector missing ho to program stop hota hai. |
| 43 | `if not camera.isOpened():` | Check karta hai webcam open hua ya nahi. |
| 44 | `print(...)` | Camera permission/index issue ka message print hota hai. |
| 45 | `raise SystemExit(1)` | Camera unavailable ho to program stop hota hai. |
| 47 | `attendance_file = ATTENDANCE_DIR / f"attendance_{datetime.now().date()}.csv"` | Aaj ki date ke naam se attendance CSV path create hota hai. |
| 48 | `marked_users = set()` | Same session me already marked users store karne ke liye set create hota hai. Set duplicate entries avoid karta hai. |
| 50 | `print("Recognition started. Press q to stop.")` | User ko program start and quit instruction milta hai. |
| 52 | `while True:` | Live recognition loop start hota hai. |
| 53 | `success, frame = camera.read()` | Camera se current frame read hota hai. |
| 55 | `if not success:` | Frame read fail check hota hai. |
| 56 | `print("Camera frame not received.")` | Frame issue message print hota hai. |
| 57 | `break` | Loop stop hota hai. |
| 59 | `gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` | Frame grayscale me convert hota hai because detector and recognizer grayscale face image use karte hain. |
| 60 | `faces = face_detector.detectMultiScale(` | Current frame me faces detect karne ka function start hota hai. |
| 61 | `gray_frame,` | Detector ko grayscale frame input milta hai. |
| 62 | `scaleFactor=1.2,` | Different face sizes handle hote hain. |
| 63 | `minNeighbors=5,` | False detections reduce hoti hain. |
| 64 | `minSize=(80, 80)` | Minimum face size define hota hai. |
| 65 | `)` | Detection function close hota hai. |
| 67 | `for (x, y, w, h) in faces:` | Har detected face par loop chalta hai. |
| 68 | `face_image = gray_frame[y:y + h, x:x + w]` | Detected face area crop hota hai. |
| 69 | `face_image = cv2.resize(face_image, FACE_SIZE)` | Crop face training ke same size me resize hota hai. |
| 70 | `user_id, confidence = recognizer.predict(face_image)` | Model face ko predict karta hai. Output user ID and confidence score hota hai. Lower confidence generally better match hota hai. |
| 72 | `if confidence < 70:` | Threshold check hota hai. Agar confidence 70 se less hai to face registered user maana jata hai. |
| 73 | `user_name = labels.get(user_id, "Unknown")` | Predicted user ID ka name labels dictionary se milta hai. Agar ID missing ho to Unknown show hota hai. |
| 74 | `display_text = f"{user_name} ({round(confidence, 2)})"` | Screen par show hone wala name and confidence text create hota hai. |
| 75 | `box_color = (0, 255, 0)` | Recognized user ke liye green box color set hota hai. |
| 77 | `if user_id not in marked_users:` | Check karta hai user already attendance marked hai ya nahi. |
| 78 | `current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")` | Current date and time readable format me create hota hai. |
| 79 | `attendance_row = pd.DataFrame(` | Attendance row DataFrame creation start hoti hai. |
| 80 | `[[user_id, user_name, current_time]],` | Attendance row me user ID, name, and time values rakhi jati hain. |
| 81 | `columns=["User ID", "Name", "Time"]` | CSV columns ke names define hote hain. |
| 82 | `)` | DataFrame creation close hoti hai. |
| 83 | `attendance_row.to_csv(` | Attendance row CSV file me save/append karne ka function start hota hai. |
| 84 | `attendance_file,` | CSV file path diya jata hai. |
| 85 | `mode="a",` | Append mode use hota hai, taaki old attendance overwrite na ho. |
| 86 | `header=not attendance_file.exists(),` | Header sirf tab write hota hai jab file new ho. Existing file me duplicate header nahi aata. |
| 87 | `index=False` | Pandas index column CSV me save nahi hota, file clean rehti hai. |
| 88 | `)` | CSV save call close hota hai. |
| 89 | `marked_users.add(user_id)` | User ID marked set me add hoti hai, taaki same session me duplicate attendance na lage. |
| 91 | `result_path = RESULTS_DIR / f"{user_name}_{datetime.now().strftime('%H%M%S')}.jpg"` | Recognition screenshot file path create hota hai. File name me time use hota hai taaki unique name bane. |
| 92 | `cv2.imwrite(str(result_path), frame)` | Current frame result image ke form me save hota hai. |
| 93 | `else:` | Agar confidence threshold pass nahi hua, to unknown user branch run hoti hai. |
| 94 | `display_text = "Unknown"` | Unknown user ke liye display text set hota hai. |
| 95 | `box_color = (0, 0, 255)` | Unknown user ke liye red box color set hota hai. |
| 97 | `cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)` | Face ke around green/red rectangle draw hota hai. |
| 98 | `cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)` | Rectangle ke upar user name/confidence ya Unknown text draw hota hai. |
| 100 | `cv2.imshow("Face Recognition Attendance", frame)` | Final processed live frame screen par show hota hai. |
| 102 | `if cv2.waitKey(1) & 0xFF == ord("q"):` | `q` key press check hota hai. |
| 103 | `break` | User quit kare to loop stop hota hai. |
| 105 | `camera.release()` | Webcam release hota hai. |
| 106 | `cv2.destroyAllWindows()` | OpenCV windows close hoti hain. |
| 108 | `print("Recognition stopped.")` | Stop message print hota hai. |
| 109 | `print(f"Attendance saved at: {attendance_file}")` | Attendance file path print hota hai. |

---

# 9. Final Run Guide

Project ko hamesha sequence me run karna hai. Pehle images capture, phir model train, phir recognition and attendance.

```bash
python3 src/01_capture_images.py
python3 src/02_train_model.py
python3 src/03_recognize_and_attendance.py
```

## Final Teaching Summary

Teacher speaking flow: "Students, aaj humne complete face recognition attendance project build kiya. Pehle OpenCV se face detect kiya. Phir webcam se user images capture ki. Phir LBPH recognizer se model train kiya. Phir live camera me user identify kiya. Recognized user ki attendance CSV me save ki aur result screenshot bhi save kiya. Ye project Computer Vision, Machine Learning, file handling, CSV reporting, and real-time automation ka practical combination hai."
