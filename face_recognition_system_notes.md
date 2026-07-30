# Project - Face Recognition System

## Topics Covered

Face Detection using OpenCV, Capture Images from Webcam, Face Recognition using Machine Learning, Train Face Recognition Model, Identify Registered Users, Real-time Face Detection, Attendance Marking Basic, and Save Recognition Results.

Is project ka main goal hai students ko ek complete real-world Computer Vision project step by step build karwana. Hum ek saath complete code nahi likhenge. Pehle topic samjhenge, phir us topic ka project part build karenge, phir code likhenge, phir code ki har line ko detail me simple Hinglish me explain karenge.

Teacher speaking flow: "Students, aaj hum Face Recognition System build karenge. Is project me webcam se face detect hoga, registered users ke face images capture honge, un images se model train hoga, phir system real-time camera me user ko identify karega, attendance mark karega, aur recognition result save karega. Hum isko part by part build karenge, taaki aapko sirf code nahi, pura project logic samajh aaye."

Important privacy note: Face recognition sensitive technology hai. Is project ko sirf learning, classroom demo, aur consent-based practice ke liye use karna chahiye. Kisi ka face data bina permission capture ya use nahi karna chahiye.

---

# 1. Face Detection using OpenCV

Face Detection ka matlab image ya video frame me face ka location find karna. Face detection sirf ye batata hai ki face kaha hai. Ye nahi batata ki face kis person ka hai. Agar camera frame me Rahul ka face hai, to face detection sirf rectangle draw karega; ye Rahul ko identify nahi karega. Identification ka kaam face recognition ka hota hai.

OpenCV ek popular Computer Vision library hai. Computer Vision ka matlab computer ko images aur videos samjhana. OpenCV se hum webcam open kar sakte hain, image frames read kar sakte hain, image ko grayscale me convert kar sakte hain, face detect kar sakte hain, rectangle draw kar sakte hain, aur live video screen par show kar sakte hain.

Is project me hum Haar Cascade face detector use karenge. Haar Cascade OpenCV ka pre-trained face detector hai. Pre-trained ka matlab ye already face patterns par trained hota hai. Hume face detection ke liye model ko manually train nahi karna. Hume sirf detector load karna hai aur webcam frame par use karna hai.

Real-world use case: Face detection CCTV monitoring, camera apps, attendance systems, entry systems, photo tagging, and video analytics me first step hota hai. Face recognition se pehle face detection zaroori hota hai.

## Project Build Part 1 - Webcam se Face Detect Karna

Is part me hum webcam open karenge, har frame read karenge, frame ko grayscale me convert karenge, face detect karenge, aur detected face ke around rectangle draw karenge. Ye project ka foundation hai.

## Practice Code 1 - Face Detection

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
Face detect hone par face ke around green rectangle show hoga.
q press karne par webcam window close ho jayegi.
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `import cv2` | Ye line OpenCV library import karti hai. OpenCV webcam access, image processing, face detection, rectangle drawing, and video display ke liye use hoti hai. Agar OpenCV import nahi hoga to project ka computer vision part start hi nahi hoga. |
| 3 | `CASCADE_PATH = ...` | Ye Haar Cascade XML file ka path banata hai. Ye file OpenCV ke andar already available hoti hai aur frontal face detect karne ke liye use hoti hai. |
| 5 | `face_detector = cv2.CascadeClassifier(CASCADE_PATH)` | Ye face detector object create karta hai. Is object ka kaam webcam frame me face locations find karna hai. |
| 6 | `camera = cv2.VideoCapture(0)` | Ye default webcam open karta hai. `0` ka matlab first/default camera. Agar external camera use ho to kabhi `1` use karna pad sakta hai. |
| 8 | `while True:` | Ye infinite loop start karta hai. Webcam video continuous frames deta hai, isliye hume baar-baar frame read karna hota hai. |
| 9 | `success, frame = camera.read()` | Ye webcam se ek frame read karta hai. `success` batata hai frame mila ya nahi, aur `frame` actual image hoti hai. |
| 11 | `if not success:` | Ye check karta hai ki frame receive hua ya nahi. Agar camera permission issue ya camera off hai to success false ho sakta hai. |
| 12 | `print(...)` | Agar frame nahi mila to user ko readable error message milta hai. |
| 13 | `break` | Loop stop kar deta hai because frame ke bina face detection possible nahi. |
| 15 | `gray_frame = cv2.cvtColor(...)` | Color frame ko grayscale me convert karta hai. Haar Cascade grayscale image par fast aur stable kaam karta hai. |
| 17 | `faces = face_detector.detectMultiScale(...)` | Ye grayscale image me faces detect karta hai. Output me face coordinates milte hain. |
| 18 | `gray_frame` | Detector ko grayscale image input di ja rahi hai. |
| 19 | `scaleFactor=1.2` | Detector image ko different sizes par check karta hai because face camera se near ya far ho sakta hai. |
| 20 | `minNeighbors=5` | Ye detection quality control karta hai. Isse false detection kam ho sakti hai. |
| 21 | `minSize=(80, 80)` | Ye minimum face size set karta hai. Bahut chhote objects ko face nahi maana jayega. |
| 24 | `for (x, y, w, h) in faces:` | Har detected face ke coordinates par loop chalata hai. `x`, `y` position hai, `w` width hai, `h` height hai. |
| 25 | `cv2.rectangle(...)` | Detected face ke around green rectangle draw karta hai. `(0, 255, 0)` green color hai aur `2` border thickness hai. |
| 27 | `cv2.imshow(...)` | Webcam frame ko screen par show karta hai. |
| 29 | `cv2.waitKey(1) ...` | Keyboard key check karta hai. Agar user `q` press kare to loop stop hoga. |
| 32 | `camera.release()` | Webcam resource release karta hai. Ye cleanup step important hai. |
| 33 | `cv2.destroyAllWindows()` | OpenCV ki windows close karta hai. |

---

# 2. Capture Images from Webcam

Face recognition model ko train karne ke liye registered users ke face images chahiye. Agar system ko Rahul ko recognize karna hai, to Rahul ke multiple face images capture karne padenge. Agar Priya ko recognize karna hai, to Priya ke images bhi capture karne padenge.

Multiple images isliye important hain kyunki face hamesha same nahi dikhta. Kabhi light change hoti hai, kabhi face angle change hota hai, kabhi expression change hota hai. Agar model ko sirf ek image milegi, to model weak learn karega. Isliye hum har user ke 50 face images capture karenge.

Real-world use case: Attendance system me pehle students/employees ko register kiya jaata hai. Registration ke time unke face images capture hote hain. Ye captured images training data ban jaate hain.

## Project Build Part 2 - Registered User ke Images Capture Karna

Is part me hum user se numeric ID aur name input lenge. Phir user ke naam se folder banayenge. Webcam se face detect hoga, face crop hoga, aur cropped face image user folder me save hogi.

## Practice Code 2 - Capture User Images

```python
import cv2
from pathlib import Path

DATASET_DIR = Path("dataset")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

DATASET_DIR.mkdir(exist_ok=True)

user_id = input("Enter numeric user ID: ").strip()
user_name = input("Enter user name: ").strip().replace(" ", "_")

user_folder = DATASET_DIR / f"{user_id}_{user_name}"
user_folder.mkdir(exist_ok=True)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
camera = cv2.VideoCapture(0)

image_count = 0
max_images = 50
```

## Output

```text
Enter numeric user ID: 1
Enter user name: Rahul
dataset/1_Rahul folder create hoga.
Webcam start hoga.
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `import cv2` | OpenCV import hota hai. Webcam, face detection, image crop, image save, and display ke liye OpenCV use hoga. |
| 2 | `from pathlib import Path` | `Path` file and folder path ko clean way me handle karta hai. Isse folder create karna aur file path banana easy hota hai. |
| 4 | `DATASET_DIR = Path("dataset")` | Dataset folder ka path define karta hai. Captured face images isi folder ke andar save hongi. |
| 5 | `CASCADE_PATH = ...` | Haar Cascade detector file ka path define karta hai. Ye face detection ke liye required hai. |
| 7 | `DATASET_DIR.mkdir(exist_ok=True)` | Dataset folder create karta hai. `exist_ok=True` ka matlab folder already exist kare to error nahi aayega. |
| 9 | `user_id = input(...).strip()` | User se numeric ID input leta hai. `strip()` extra spaces remove karta hai. Ye ID training label ke roop me use hogi. |
| 10 | `user_name = input(...).strip().replace(" ", "_")` | User ka name input leta hai. Space ko underscore me convert karta hai taaki folder name clean rahe. |
| 12 | `user_folder = DATASET_DIR / f"{user_id}_{user_name}"` | User ke liye folder path banata hai. Example: `dataset/1_Rahul`. |
| 13 | `user_folder.mkdir(exist_ok=True)` | User ka folder create karta hai. Isi folder me us user ke face images save hongi. |
| 15 | `face_detector = cv2.CascadeClassifier(CASCADE_PATH)` | Haar Cascade face detector load karta hai. |
| 16 | `camera = cv2.VideoCapture(0)` | Webcam start karta hai. |
| 18 | `image_count = 0` | Captured images count karne ke liye counter start karta hai. |
| 19 | `max_images = 50` | Maximum images limit set karta hai. Jab 50 images capture ho jayengi to capture stop hoga. |

## Practice Code 3 - Capture Loop

```python
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
        image_path = user_folder / f"{image_count}.jpg"
        cv2.imwrite(str(image_path), face_image)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Capture Face Images", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if image_count >= max_images:
        break

camera.release()
cv2.destroyAllWindows()
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `while True:` | Continuous webcam loop start karta hai. Camera baar-baar frames deta hai. |
| 2 | `success, frame = camera.read()` | Webcam se current frame read karta hai. |
| 4 | `if not success:` | Check karta hai ki frame mila ya nahi. |
| 5 | `print(...)` | Agar frame nahi mila to message show karta hai. |
| 6 | `break` | Loop stop karta hai. |
| 8 | `gray_frame = cv2.cvtColor(...)` | Frame ko grayscale me convert karta hai. |
| 9 | `faces = face_detector.detectMultiScale(...)` | Frame me face locations detect karta hai. |
| 17 | `for (x, y, w, h) in faces:` | Har detected face par loop chalata hai. |
| 18 | `image_count += 1` | Captured image counter increase karta hai. |
| 20 | `face_image = gray_frame[y:y + h, x:x + w]` | Face region crop karta hai. Full frame nahi, sirf face image save hogi. |
| 21 | `image_path = user_folder / f"{image_count}.jpg"` | Image save karne ka file path create karta hai. |
| 22 | `cv2.imwrite(str(image_path), face_image)` | Cropped face image ko JPG file me save karta hai. |
| 24 | `cv2.rectangle(...)` | Face ke around rectangle draw karta hai taaki user ko detection dikhe. |
| 26 | `cv2.imshow(...)` | Webcam output screen par show karta hai. |
| 28 | `cv2.waitKey(...)` | User `q` press kare to capture stop hota hai. |
| 31 | `if image_count >= max_images:` | Check karta hai kya 50 images capture ho chuki hain. |
| 32 | `break` | Max images complete hone par loop stop karta hai. |
| 34 | `camera.release()` | Webcam release karta hai. |
| 35 | `cv2.destroyAllWindows()` | OpenCV windows close karta hai. |

---

# 3. Face Recognition using Machine Learning

Face Recognition ka matlab detected face ko identify karna. Face Detection batata hai face kaha hai, Face Recognition batata hai face kis registered user ka hai. Recognition ke liye model ko training images chahiye hoti hain.

Is project me hum OpenCV ka LBPH Face Recognizer use karenge. LBPH ka full form Local Binary Patterns Histograms hai. Simple words me, ye face ke local texture patterns ko numbers me convert karta hai. Jab new face aata hai, model us pattern ko trained faces ke patterns se compare karta hai.

Real-world use case: Face recognition attendance system, access control system, employee verification, smart door lock, and lab entry system me use ho sakta hai.

## Project Build Part 3 - Training Data Read Karna

Is part me hum `dataset/` folder se captured images read karenge. Folder name se user ID aur user name nikalenge. Images ko `faces` list me aur user IDs ko `labels` list me store karenge.

## Practice Code 4 - Prepare Training Data

```python
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
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `import cv2` | OpenCV import karta hai. LBPH face recognizer OpenCV contrib package se use hoga. |
| 2 | `import numpy as np` | NumPy import karta hai. Labels ko numeric array me convert karne ke liye use hoga. |
| 3 | `from pathlib import Path` | Folder/file paths handle karne ke liye Path import hota hai. |
| 5 | `DATASET_DIR = Path("dataset")` | Captured face images folder ka path define karta hai. |
| 6 | `TRAINER_DIR = Path("trainer")` | Trained model save karne wale folder ka path define karta hai. |
| 7 | `MODEL_PATH = TRAINER_DIR / "face_model.yml"` | Trained model file ka path define karta hai. |
| 8 | `LABELS_PATH = TRAINER_DIR / "labels.txt"` | User ID-name mapping save karne wali labels file ka path define karta hai. |
| 10 | `TRAINER_DIR.mkdir(exist_ok=True)` | Trainer folder create karta hai agar exist nahi karta. |
| 12 | `recognizer = cv2.face.LBPHFaceRecognizer_create()` | LBPH face recognizer object create karta hai. Ye model training and prediction ke liye use hoga. |
| 14 | `faces = []` | Face images store karne ke liye empty list banata hai. |
| 15 | `labels = []` | User IDs store karne ke liye empty list banata hai. |
| 16 | `label_names = {}` | User ID aur user name mapping ke liye dictionary banata hai. |

---

# 4. Train Face Recognition Model

Training ka matlab model ko examples se learning karwana. Humare examples hain captured face images. Har image ke saath user ID label hoga. Model face image aur user ID ka relation learn karega.

Training ke baad model file save hogi: `trainer/face_model.yml`. Labels file bhi save hogi: `trainer/labels.txt`. Recognition ke time model file aur labels file dono load honge.

## Project Build Part 4 - Images Read Karke Model Train Karna

## Practice Code 5 - Read Images and Train Model

```python
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

recognizer.train(faces, np.array(labels))
recognizer.write(str(MODEL_PATH))
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `for user_folder in DATASET_DIR.iterdir():` | Dataset folder ke andar har user folder par loop chalata hai. |
| 2 | `if not user_folder.is_dir():` | Check karta hai current item folder hai ya nahi. |
| 3 | `continue` | Agar folder nahi hai to skip karta hai. |
| 5 | `folder_parts = user_folder.name.split("_", 1)` | Folder name ko ID aur name me split karta hai. Example: `1_Rahul`. |
| 6 | `user_id = int(folder_parts[0])` | Folder name ka first part numeric ID me convert karta hai. |
| 7 | `user_name = ...` | Folder name se user name nikalta hai. Agar name missing ho to default name banata hai. |
| 8 | `label_names[user_id] = user_name` | User ID aur name mapping dictionary me store karta hai. |
| 10 | `for image_path in user_folder.glob("*.jpg"):` | User folder ke andar JPG images par loop chalata hai. |
| 11 | `face_image = cv2.imread(..., cv2.IMREAD_GRAYSCALE)` | Face image ko grayscale format me read karta hai. |
| 13 | `if face_image is None:` | Check karta hai agar image read nahi hui. |
| 14 | `continue` | Invalid image skip karta hai. |
| 16 | `faces.append(face_image)` | Face image training list me add karta hai. |
| 17 | `labels.append(user_id)` | Corresponding user ID labels list me add karta hai. |
| 19 | `recognizer.train(faces, np.array(labels))` | Model ko face images aur labels se train karta hai. |
| 20 | `recognizer.write(str(MODEL_PATH))` | Trained model file me save karta hai. |

---

# 5. Identify Registered Users

Registered user wo user hota hai jiska face images dataset me hain aur model us user par trained hai. Jab registered user camera ke saamne aata hai, recognizer user ID predict karta hai. User ID se hum user ka name labels file se find karte hain.

LBPH recognizer confidence score return karta hai. Lower confidence usually better match hota hai. Is project me `confidence < 70` ko recognized condition maanenge. Real project me threshold test karke adjust hota hai.

## Project Build Part 5 - Model and Labels Load Karna

## Practice Code 6 - Load Model and Labels

```python
labels = {}

with LABELS_PATH.open("r") as file:
    for line in file:
        user_id, user_name = line.strip().split(",", 1)
        labels[int(user_id)] = user_name

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))

face_detector = cv2.CascadeClassifier(CASCADE_PATH)
camera = cv2.VideoCapture(0)
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `labels = {}` | Empty dictionary create karta hai. Isme user ID aur user name mapping store hogi. |
| 3 | `with LABELS_PATH.open("r") as file:` | Labels file read mode me open karta hai. |
| 4 | `for line in file:` | Labels file ki har line read karta hai. |
| 5 | `user_id, user_name = line.strip().split(",", 1)` | Line ko user ID aur user name me split karta hai. |
| 6 | `labels[int(user_id)] = user_name` | User ID ko integer me convert karke dictionary me name ke saath store karta hai. |
| 8 | `recognizer = cv2.face.LBPHFaceRecognizer_create()` | LBPH recognizer object create karta hai. |
| 9 | `recognizer.read(str(MODEL_PATH))` | Trained model file load karta hai. |
| 11 | `face_detector = cv2.CascadeClassifier(CASCADE_PATH)` | Face detector load karta hai. |
| 12 | `camera = cv2.VideoCapture(0)` | Webcam start karta hai. |

---

# 6. Real-time Face Detection and Recognition

Real-time ka matlab live webcam frames par continuously process karna. Webcam continuously frames deta hai. Har frame me face detect hoga, detected face crop hoga, model prediction karega, aur output me recognized user ka name ya Unknown show hoga.

## Project Build Part 6 - Live Recognition

## Practice Code 7 - Recognize Face in Real Time

```python
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
        else:
            display_text = "Unknown"
            box_color = (0, 0, 255)
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `while True:` | Continuous webcam loop start karta hai. |
| 2 | `success, frame = camera.read()` | Camera se current frame read karta hai. |
| 4 | `if not success:` | Check karta hai frame mila ya nahi. |
| 5 | `print(...)` | Agar frame nahi mila to message print karta hai. |
| 6 | `break` | Loop stop karta hai. |
| 8 | `gray_frame = cv2.cvtColor(...)` | Frame ko grayscale me convert karta hai. |
| 9 | `faces = face_detector.detectMultiScale(...)` | Frame me faces detect karta hai. |
| 17 | `for (x, y, w, h) in faces:` | Har detected face ke coordinates par loop chalata hai. |
| 18 | `face_image = gray_frame[y:y + h, x:x + w]` | Detected face crop karta hai. |
| 19 | `user_id, confidence = recognizer.predict(face_image)` | Model cropped face ko predict karta hai aur user ID + confidence return karta hai. |
| 21 | `if confidence < 70:` | Confidence threshold check karta hai. Lower confidence better match maana jaata hai. |
| 22 | `user_name = labels.get(user_id, "Unknown")` | User ID se name find karta hai. Agar name nahi milta to Unknown. |
| 23 | `display_text = ...` | Screen par show hone wala text create karta hai. |
| 24 | `box_color = (0, 255, 0)` | Recognized face ke liye green color set karta hai. |
| 25 | `else:` | Agar confidence threshold pass nahi hua to unknown condition chalegi. |
| 26 | `display_text = "Unknown"` | Unknown face ke liye text set karta hai. |
| 27 | `box_color = (0, 0, 255)` | Unknown face ke liye red color set karta hai. |

---

# 7. Attendance Marking Basic

Attendance marking ka matlab recognized user ka record CSV file me save karna. Hum user ID, name, aur current time save karenge. Same session me duplicate attendance avoid karne ke liye `marked_users` set use hoga.

## Project Build Part 7 - Attendance Save Karna

## Practice Code 8 - Mark Attendance

```python
attendance_file = ATTENDANCE_DIR / f"attendance_{datetime.now().date()}.csv"
marked_users = set()

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
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `attendance_file = ...` | Current date ke naam se attendance CSV file path banata hai. |
| 2 | `marked_users = set()` | Already marked users ko store karne ke liye set banata hai. |
| 4 | `if user_id not in marked_users:` | Check karta hai user ki attendance already marked hai ya nahi. |
| 5 | `current_time = datetime.now().strftime(...)` | Current date and time readable format me convert karta hai. |
| 6 | `attendance_row = pd.DataFrame(...)` | Attendance row ko table/DataFrame form me banata hai. |
| 7 | `[[user_id, user_name, current_time]]` | One row ka actual attendance data hai. |
| 8 | `columns=[...]` | CSV ke column names define karta hai. |
| 10 | `attendance_row.to_csv(...)` | Attendance row ko CSV file me save karta hai. |
| 12 | `mode="a"` | Append mode use karta hai, taaki new row file ke end me add ho. |
| 13 | `header=not attendance_file.exists()` | Agar file first time ban rahi hai to header add hoga, warna repeat nahi hoga. |
| 14 | `index=False` | Extra DataFrame index CSV me save nahi hota. |
| 16 | `marked_users.add(user_id)` | User ko marked list me add karta hai taaki duplicate attendance na ho. |

---

# 8. Save Recognition Results

Recognition results save karna useful hota hai because later proof ke form me screenshot dekha ja sakta hai. Jab registered user recognize hota hai, system current webcam frame ko image file ke form me `results/` folder me save kar sakta hai.

## Project Build Part 8 - Recognition Screenshot Save Karna

## Practice Code 9 - Save Result Image

```python
result_path = RESULTS_DIR / f"{user_name}_{datetime.now().strftime('%H%M%S')}.jpg"
cv2.imwrite(str(result_path), frame)
```

## Detailed Code Explanation

| Line | Code | Explanation |
|---|---|---|
| 1 | `result_path = ...` | Result image ka file path create karta hai. File name me user name aur current time include hota hai. |
| 1 | `datetime.now().strftime('%H%M%S')` | Current hour, minute, second ko string me convert karta hai. Isse file name unique banta hai. |
| 2 | `cv2.imwrite(str(result_path), frame)` | Current webcam frame ko image file ke form me save karta hai. |

---

# Final Complete Project Flow

Project ko build karne ka correct order:

1. `python3 src/01_capture_images.py`
2. User ID aur name enter karo.
3. Webcam se face images capture karo.
4. `python3 src/02_train_model.py`
5. Captured images se model train karo.
6. `python3 src/03_recognize_and_attendance.py`
7. Webcam me registered user ko identify karo.
8. Attendance CSV file me save hogi.
9. Recognition screenshot `results/` folder me save hoga.

## Final Summary

Is project me students ne topic by topic Face Recognition System build kiya. Pehle Face Detection using OpenCV samjha, phir webcam se images capture ki, phir Machine Learning model train kiya, phir registered users identify kiye, phir real-time recognition kiya, attendance mark ki, aur recognition result save kiya.

Teacher speaking flow: "Students, ab aapne dekha ki Face Recognition project ek single code file ka naam nahi hai. Ye multiple small steps ka combination hai. Detection, data collection, training, recognition, attendance, and result saving sab connected hain. Agar aap ye flow samajh gaye, to aap real-world Computer Vision projects ka foundation samajh gaye."
