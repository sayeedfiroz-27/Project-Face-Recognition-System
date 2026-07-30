# Project - Face Recognition System

## Topics Covered

Face Detection using OpenCV, Capture Images from Webcam, Face Recognition using Machine Learning, Train Face Recognition Model, Identify Registered Users, Real-time Face Detection, Attendance Marking Basic, and Save Recognition Results.

Is project ka main goal hai students ko ek complete real-world Computer Vision project step by step build karwana. Hum ek saath complete code nahi likhenge. Pehle topic samjhenge, phir us topic ka project part build karenge, phir code likhenge, phir code ki har line ko detail me simple Hinglish me explain karenge.

Teacher speaking flow: "Students, aaj hum Face Recognition System build karenge. Is project me webcam se face detect hoga, registered users ke face images capture honge, un images se model train hoga, phir system real-time camera me user ko identify karega, attendance mark karega, aur recognition result save karega. Hum isko part by part build karenge, taaki aapko sirf code nahi, pura project logic samajh aaye."

Important privacy note: Face recognition sensitive technology hai. Is project ko sirf learning, classroom demo, aur consent-based practice ke liye use karna chahiye. Kisi ka face data bina permission capture ya use nahi karna chahiye.

---

# Important Project Paths and Images

Is project me folders ka role bahut important hai, because Face Recognition project sirf Python code se complete nahi hota. Isme captured images, trained model, attendance file, aur saved result images bhi generate hote hain. Students ko pehle folder structure samjhana zaroori hai, taaki jab project run ho to unko clearly pata rahe ki kaunsi file kaha create hogi.

`dataset/` folder me students ke face images save honge. Jab hum `python3 src/01_capture_images.py` run karenge, script user ID aur user name poochegi. Agar user ID `1` aur name `Rahul` diya, to folder path `dataset/1_Rahul/` banega. Is folder ke andar `1.jpg`, `2.jpg`, `3.jpg` jaise multiple face images save honge. GitHub me real face images include nahi ki gayi hain, because face data private hota hai. Classroom practical ke time students apni consent-based images capture karenge.

`trainer/` folder training ke baad files store karega. Jab `python3 src/02_train_model.py` run hoga, to `trainer/face_model.yml` file create hogi. Ye trained model file hai. Iske saath `trainer/labels.txt` file create hogi, jisme user ID aur user name ka relation save hota hai. Example: `1,Rahul`. Recognition ke time model user ID predict karta hai, aur labels file se us ID ka name milta hai.

`attendance/` folder me CSV attendance files save hongi. Jab recognition script run hogi aur registered user recognize hoga, to `attendance/attendance_YYYY-MM-DD.csv` file me user ID, name, aur time save hoga. Iska fayda ye hai ki teacher baad me attendance report open karke dekh sakta hai.

`results/` folder me recognition screenshots save honge. Jab registered user successfully identify hota hai, system current camera frame ko image ke form me save karta hai. Example path `results/Rahul_143025.jpg` ho sakta hai. Ye proof ke form me useful hota hai ki kis user ko kis time recognize kiya gaya.

Teacher speaking flow: "Students, dhyaan rakho, GitHub me folder structure diya gaya hai, lekin real face images privacy ke reason se nahi di gayi. Images project run karte waqt webcam se generate hongi. Aapko bas sequence follow karna hai: pehle capture, phir train, phir recognize."

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

---

# Complete Source Code Explanation - Teacher Reading Section

Ye section actual GitHub code ke according hai. Isko aap class me direct read kar sakte ho. Yaha har file ka purpose, important lines ka meaning, keywords ka use, aur real project me line ka kaam simple Hinglish me explain kiya gaya hai.

## File 1 - `src/01_capture_images.py`

Is file ka kaam registered user ke face images capture karna hai. Face Recognition model ko kisi person ko pehchanna sikhane ke liye us person ke multiple face examples chahiye. Jaise student ko kisi new person ko identify karna ho to woh us person ko alag-alag angle aur lighting me dekhkar better pehchan pata hai, waise hi model ko bhi multiple face images chahiye. Ye script webcam open karti hai, face detect karti hai, face crop karti hai, same size me resize karti hai, aur `dataset/` folder me save karti hai.

```python
import cv2
from pathlib import Path

DATASET_DIR = Path("dataset")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (200, 200)
```

`import cv2` OpenCV library ko import karta hai. OpenCV camera open karne, image process karne, face detect karne, rectangle draw karne, aur image save karne ke liye use hoti hai. Agar ye line nahi hogi to `cv2.VideoCapture`, `cv2.CascadeClassifier`, `cv2.imwrite`, aur `cv2.imshow` use nahi ho paayenge.

`from pathlib import Path` file aur folder paths ko clean way me handle karne ke liye use hota hai. `Path("dataset")` ek path object banata hai. Path object ka fayda ye hai ki hum `DATASET_DIR / "1_Rahul"` jaise readable syntax se folder path bana sakte hain.

`DATASET_DIR = Path("dataset")` batata hai ki captured face images `dataset/` folder me save hongi. Ye variable project ka main image storage location define karta hai. Variable banane ka fayda ye hai ki agar future me folder name change karna ho to sirf ek jagah change karna padega.

`CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"` OpenCV ke built-in face detector XML file ka path banata hai. `cv2.data.haarcascades` OpenCV ka default cascade folder deta hai, aur XML file frontal face detect karne ke liye pre-trained hoti hai.

`FACE_SIZE = (200, 200)` har cropped face ko 200 by 200 pixels me resize karne ke liye use hota hai. Training aur prediction me same size images dena model ke liye helpful hota hai. Agar images different sizes me hongi to model inconsistent input dekhega; fixed size se project stable hota hai.

```python
DATASET_DIR.mkdir(exist_ok=True)

user_id = input("Enter numeric user ID: ").strip()
user_name = input("Enter user name: ").strip().replace(" ", "_")

user_folder = DATASET_DIR / f"{user_id}_{user_name}"
user_folder.mkdir(exist_ok=True)
```

`DATASET_DIR.mkdir(exist_ok=True)` dataset folder create karta hai. `exist_ok=True` ka matlab agar folder already exist karta hai to error mat do. Classroom me script multiple times run ho sakti hai, isliye ye line practical aur safe hai.

`input("Enter numeric user ID: ")` terminal par user se ID leta hai. Model internally names ke bajaye numeric labels ke saath train hota hai, isliye numeric user ID important hai. `.strip()` extra spaces remove karta hai taaki accidental spaces problem create na karein.

`input("Enter user name: ")` user ka readable name leta hai. `.replace(" ", "_")` spaces ko underscore me convert karta hai, jaise `Rahul Sharma` ko `Rahul_Sharma`. File/folder names me underscore safer aur cleaner hota hai.

`user_folder = DATASET_DIR / f"{user_id}_{user_name}"` user-specific folder path banata hai. Example: `dataset/1_Rahul`. Is naming pattern se training ke time folder name se ID aur name dono easily mil jate hain.

`user_folder.mkdir(exist_ok=True)` selected user ke liye folder create karta hai. Agar folder already hai to usi folder me new images save ho sakti hain. Ye useful hai jab kisi user ke liye extra images capture karni ho.

```python
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
camera = cv2.VideoCapture(0)

if face_detector.empty():
    print("Face detector could not be loaded. Please check OpenCV installation.")
    raise SystemExit(1)

if not camera.isOpened():
    print("Camera could not be opened. Please check webcam permission or camera index.")
    raise SystemExit(1)
```

`cv2.CascadeClassifier(CASCADE_PATH)` Haar Cascade XML ko load karke face detector object banata hai. Ye object baad me image frame me face ka location find karega. Agar XML path wrong hai ya OpenCV install incomplete hai, detector load nahi hoga.

`cv2.VideoCapture(0)` default webcam open karta hai. `0` ka matlab first/default camera. Laptop webcam normally `0` hota hai. Agar external webcam use ho to kabhi `1` try karna pad sakta hai.

`face_detector.empty()` check karta hai ki detector properly load hua ya nahi. Ye beginner-friendly safety check hai. Agar detector load nahi hua to code aage nahi badhega, warna face detection silent fail ho sakta tha.

`raise SystemExit(1)` program ko clean error ke saath stop karta hai. Ye crash nahi, controlled stop hai. Student ko clear message milta hai ki setup issue fix karna hai.

`camera.isOpened()` check karta hai ki webcam successfully open hua ya nahi. Agar camera permission off hai, camera busy hai, ya index wrong hai to ye false hoga. Isse class me debugging easy hoti hai.

```python
image_count = 0
max_images = 50

while True:
    success, frame = camera.read()
```

`image_count = 0` saved images ka counter start karta hai. Jab bhi ek face image save hogi, count increase hoga. Is count se file names `1.jpg`, `2.jpg`, `3.jpg` bante hain.

`max_images = 50` decide karta hai ki ek user ke liye 50 images capture karni hain. Face recognition me 1 image enough nahi hoti. 50 images se model ko face ke thode variations milte hain, jisse recognition better hota hai.

`while True:` infinite loop start karta hai. Webcam video continuous frames ka sequence hota hai, isliye hume repeatedly frame read karna padta hai.

`success, frame = camera.read()` camera se ek current frame read karta hai. `success` batata hai frame mila ya nahi, aur `frame` actual image data hota hai.

```python
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(
    gray_frame,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(80, 80)
)
```

`cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` color frame ko grayscale me convert karta hai. Haar Cascade face detector grayscale image par fast aur stable kaam karta hai. Color information face detection ke liye compulsory nahi hoti.

`detectMultiScale()` frame ke andar faces detect karta hai. Output me `(x, y, w, h)` values milti hain. `x` aur `y` face box ka starting point hai, `w` width hai, aur `h` height hai.

`scaleFactor=1.2` detector ko different face sizes check karne me help karta hai. Face camera ke paas hai to bada dikhega, door hai to chhota dikhega. Scale factor different sizes handle karta hai.

`minNeighbors=5` false detection kam karta hai. Simple words me, detector ko face confirm karne ke liye nearby repeated evidence chahiye. Value bahut low hogi to false faces aa sakte hain, bahut high hogi to real faces miss ho sakte hain.

`minSize=(80, 80)` 80x80 pixels se chhote areas ignore karta hai. Isse background noise ya chhote patterns face detect hone se bachte hain.

```python
for (x, y, w, h) in faces:
    image_count += 1
    face_image = gray_frame[y:y + h, x:x + w]
    face_image = cv2.resize(face_image, FACE_SIZE)
    image_path = user_folder / f"{image_count}.jpg"
    cv2.imwrite(str(image_path), face_image)
```

`for (x, y, w, h) in faces:` har detected face par loop chalata hai. Agar ek face detected hai to loop ek baar chalega. Agar multiple faces hain to har face process hoga.

`image_count += 1` saved image counter increase karta hai. Ye line file naming aur progress tracking dono ke liye important hai.

`gray_frame[y:y + h, x:x + w]` full camera frame se sirf face region crop karta hai. `y:y+h` vertical area select karta hai aur `x:x+w` horizontal area select karta hai. Ye background remove karke only face data save karta hai.

`cv2.resize(face_image, FACE_SIZE)` cropped face image ko 200x200 fixed size me convert karta hai. Training ke liye same size images better hoti hain.

`image_path = user_folder / f"{image_count}.jpg"` current image ka path banata hai. Example: `dataset/1_Rahul/12.jpg`.

`cv2.imwrite(str(image_path), face_image)` cropped face ko JPG file ke form me save karta hai. `str(image_path)` isliye diya hai kyunki OpenCV string path expect karta hai.

```python
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(frame, f"Images: {image_count}/{max_images}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Capture Face Images", frame)
```

`cv2.rectangle()` live frame par green box draw karta hai. Isse student ko screen par dikhta hai ki system face correctly detect kar raha hai.

`cv2.putText()` rectangle ke upar progress text show karta hai, jaise `Images: 20/50`. Isse user ko pata chalta hai capture kitna complete hua.

`cv2.imshow()` webcam window show karta hai. Window ke bina code background me run hoga, lekin student output nahi dekh paayega.

```python
if cv2.waitKey(1) & 0xFF == ord("q"):
    break

if image_count >= max_images:
    break

camera.release()
cv2.destroyAllWindows()
```

`cv2.waitKey(1)` keyboard input check karta hai. Agar `q` press hota hai to loop stop hota hai. Ye manual stop option hai.

`image_count >= max_images` check karta hai ki required images complete ho gayi ya nahi. 50 images complete hote hi script automatically stop ho jaati hai.

`camera.release()` webcam ko free karta hai. Agar ye nahi karenge to camera next script ke liye busy reh sakta hai.

`cv2.destroyAllWindows()` OpenCV windows close karta hai. Ye clean shutdown step hai.

## File 2 - `src/02_train_model.py`

Is file ka kaam captured images se face recognition model train karna hai. First script ne `dataset/` me images save ki. Ab ye script un images ko read karegi, folder name se user ID aur name samjhegi, images ko labels ke saath model ko degi, aur trained model ko `trainer/face_model.yml` me save karegi.

```python
import cv2
import numpy as np
from pathlib import Path
```

`cv2` image read karne aur LBPH face recognizer create karne ke liye use hota hai. `numpy` labels ko numeric array me convert karne ke liye use hota hai. `Path` folder/file paths handle karne ke liye use hota hai.

```python
DATASET_DIR = Path("dataset")
TRAINER_DIR = Path("trainer")
MODEL_PATH = TRAINER_DIR / "face_model.yml"
LABELS_PATH = TRAINER_DIR / "labels.txt"
FACE_SIZE = (200, 200)
```

`DATASET_DIR` training images ka source folder hai. `TRAINER_DIR` output folder hai. `MODEL_PATH` trained model file ka path hai. `LABELS_PATH` ID-name mapping file ka path hai. `FACE_SIZE` ensure karta hai ki training images same size me model ko milein.

```python
TRAINER_DIR.mkdir(exist_ok=True)

if not DATASET_DIR.exists():
    print("Dataset folder not found. Please run 01_capture_images.py first.")
    raise SystemExit(1)

if not hasattr(cv2, "face"):
    print("cv2.face module not found. Please install opencv-contrib-python.")
    raise SystemExit(1)
```

`TRAINER_DIR.mkdir(exist_ok=True)` trainer folder create karta hai. Model aur labels files isi folder me save hongi.

`if not DATASET_DIR.exists():` check karta hai ki dataset folder hai ya nahi. Agar capture step run nahi hua to training possible nahi hai, isliye code direct clear message deta hai.

`if not hasattr(cv2, "face"):` check karta hai ki OpenCV ka face recognition module available hai ya nahi. LBPH recognizer `opencv-contrib-python` package me hota hai. Agar ye missing hai to install command run karna hoga.

```python
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_names = {}
```

`LBPHFaceRecognizer_create()` LBPH model object banata hai. LBPH face ke local texture patterns ko read karta hai aur un patterns ko user IDs se connect karna seekhta hai.

`faces = []` me training images store hongi. `labels = []` me har image ka correct user ID store hoga. `label_names = {}` me user ID aur user name ka mapping store hoga.

```python
for user_folder in DATASET_DIR.iterdir():
    if not user_folder.is_dir():
        continue

    folder_parts = user_folder.name.split("_", 1)
    user_id = int(folder_parts[0])
    user_name = folder_parts[1] if len(folder_parts) > 1 else f"User_{user_id}"
    label_names[user_id] = user_name
```

`DATASET_DIR.iterdir()` dataset folder ke andar har item read karta hai. Har user ke images separate folder me hote hain, jaise `1_Rahul`.

`if not user_folder.is_dir(): continue` non-folder files skip karta hai. Ye code ko safe banata hai.

`split("_", 1)` folder name ko two parts me divide karta hai: ID aur name. Example `1_Rahul` se ID `1` aur name `Rahul` milta hai.

`int(folder_parts[0])` ID ko integer me convert karta hai, because model numeric labels expect karta hai.

`label_names[user_id] = user_name` user mapping save karta hai. Baad me ye mapping `labels.txt` me write hogi.

```python
for image_path in user_folder.glob("*.jpg"):
    face_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if face_image is None:
        continue

    face_image = cv2.resize(face_image, FACE_SIZE)
    faces.append(face_image)
    labels.append(user_id)
```

`glob("*.jpg")` current user folder ke andar sirf JPG images select karta hai. Ye captured face images hoti hain.

`cv2.imread(..., cv2.IMREAD_GRAYSCALE)` image ko grayscale me read karta hai. LBPH ke liye grayscale face enough hota hai.

`if face_image is None:` unreadable/corrupt image check karta hai. Agar image read nahi hui to code us image ko skip karta hai.

`cv2.resize()` image ko 200x200 me convert karta hai. `faces.append()` image ko training list me add karta hai. `labels.append(user_id)` us image ka correct answer add karta hai.

```python
if len(faces) == 0:
    print("No training images found. Please run 01_capture_images.py first.")
    raise SystemExit(1)

recognizer.train(faces, np.array(labels))
recognizer.write(str(MODEL_PATH))
```

`len(faces) == 0` check karta hai ki images mili ya nahi. Agar images nahi mili to model train nahi hoga.

`recognizer.train(faces, np.array(labels))` main training line hai. `faces` input data hai aur labels correct answers hain. Model in examples se learn karta hai ki kaunsa face pattern kis user ID se related hai.

`recognizer.write(str(MODEL_PATH))` trained model file save karta hai. Is saved model ko third script load karegi.

```python
with LABELS_PATH.open("w") as file:
    for user_id, user_name in label_names.items():
        file.write(f"{user_id},{user_name}\n")
```

`LABELS_PATH.open("w")` labels file write mode me open karta hai. `with` file ko automatically close karta hai. Loop har user ID aur name ko file me save karta hai. Example line: `1,Rahul`. Ye file recognition ke time ID ko name me convert karne ke liye required hai.

## File 3 - `src/03_recognize_and_attendance.py`

Is file ka kaam trained model load karke live camera me user identify karna, attendance mark karna, aur recognition result image save karna hai. Ye final real-world use case hai. Yahi part students ko dikhata hai ki Machine Learning model training ke baad actual input par prediction kaise karta hai.

```python
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
```

`cv2` webcam, face detection, prediction, rectangle, text, aur image saving ke liye use hota hai. `pandas` attendance row ko DataFrame bana kar CSV me save karta hai. `datetime` current date and time nikalta hai. `Path` folders aur file paths manage karta hai.

```python
TRAINER_DIR = Path("trainer")
ATTENDANCE_DIR = Path("attendance")
RESULTS_DIR = Path("results")
MODEL_PATH = TRAINER_DIR / "face_model.yml"
LABELS_PATH = TRAINER_DIR / "labels.txt"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_SIZE = (200, 200)
```

`TRAINER_DIR` trained model location hai. `ATTENDANCE_DIR` CSV report location hai. `RESULTS_DIR` screenshots location hai. `MODEL_PATH` model file ka exact path hai. `LABELS_PATH` user ID-name mapping file hai. `CASCADE_PATH` face detector file hai. `FACE_SIZE` prediction ke face crop ko training ke same size me convert karta hai.

```python
ATTENDANCE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    print("Trained model or labels file not found.")
    print("Please run 02_train_model.py first.")
    raise SystemExit(1)
```

`ATTENDANCE_DIR.mkdir()` aur `RESULTS_DIR.mkdir()` output folders create karte hain. Agar folders already exist karte hain to error nahi aata.

`if not MODEL_PATH.exists() or not LABELS_PATH.exists():` training output check karta hai. Agar model ya labels missing hain to recognition run nahi ho sakta. Isliye code student ko pehle training script run karne ka message deta hai.

```python
labels = {}

with LABELS_PATH.open("r") as file:
    for line in file:
        user_id, user_name = line.strip().split(",", 1)
        labels[int(user_id)] = user_name
```

`labels = {}` empty dictionary hai. Ye ID ko name se map karegi.

`LABELS_PATH.open("r")` labels file read karta hai. Har line me ID aur name comma separated hota hai.

`line.strip().split(",", 1)` newline remove karke line ko ID aur name me split karta hai. `labels[int(user_id)] = user_name` mapping create karta hai, jaise `labels[1] = "Rahul"`.

```python
if not hasattr(cv2, "face"):
    print("cv2.face module not found. Please install opencv-contrib-python.")
    raise SystemExit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(str(MODEL_PATH))
```

`hasattr(cv2, "face")` check karta hai ki LBPH module installed hai ya nahi. Ye project `opencv-contrib-python` par depend karta hai.

`LBPHFaceRecognizer_create()` recognizer object banata hai. `recognizer.read()` saved trained model ko memory me load karta hai. Ab model live face predict karne ke liye ready hai.

```python
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
camera = cv2.VideoCapture(0)

if face_detector.empty():
    print("Face detector could not be loaded. Please check OpenCV installation.")
    raise SystemExit(1)

if not camera.isOpened():
    print("Camera could not be opened. Please check webcam permission or camera index.")
    raise SystemExit(1)
```

`face_detector` live frame me face location find karega. Recognition se pehle face detection zaroori hai, because model ko cropped face image chahiye.

`camera` webcam open karta hai. Safety checks detector aur camera issue ko early catch karte hain. Ye practical class ke liye important hai because students ko exact issue pata chalna chahiye.

```python
attendance_file = ATTENDANCE_DIR / f"attendance_{datetime.now().date()}.csv"
marked_users = set()
```

`attendance_file` current date ke naam se CSV path banata hai. Example: `attendance/attendance_2026-07-30.csv`. Date-wise file real attendance systems me useful hoti hai.

`marked_users = set()` duplicate attendance avoid karta hai. Agar Rahul camera ke saamne 20 seconds khada hai, to attendance 20 baar mark nahi honi chahiye. Set me Rahul ka ID add ho jayega aur repeat entry avoid hogi.

```python
while True:
    success, frame = camera.read()

    if not success:
        print("Camera frame not received.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

`while True` live processing loop hai. `camera.read()` har frame read karta hai. Agar frame nahi mila to loop stop hota hai. `cv2.cvtColor()` frame ko grayscale banata hai, because detector aur LBPH grayscale input par kaam karte hain.

```python
faces = face_detector.detectMultiScale(
    gray_frame,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(80, 80)
)
```

`detectMultiScale()` live frame me face boxes find karta hai. `scaleFactor` different face sizes handle karta hai, `minNeighbors` false detections kam karta hai, aur `minSize` very small noise ignore karta hai.

```python
for (x, y, w, h) in faces:
    face_image = gray_frame[y:y + h, x:x + w]
    face_image = cv2.resize(face_image, FACE_SIZE)
    user_id, confidence = recognizer.predict(face_image)
```

`for` loop har detected face ko process karta hai. `face_image` crop line full frame se only face area nikalti hai. `resize` prediction input ko training size ke same banata hai. `recognizer.predict()` model se answer leta hai: predicted `user_id` aur `confidence`.

Important point: LBPH me lower confidence value better match hoti hai. Agar confidence low hai to face trained user se zyada match kar raha hai. Agar confidence high hai to match weak ho sakta hai.

```python
if confidence < 70:
    user_name = labels.get(user_id, "Unknown")
    display_text = f"{user_name} ({round(confidence, 2)})"
    box_color = (0, 255, 0)
else:
    display_text = "Unknown"
    box_color = (0, 0, 255)
```

`confidence < 70` threshold condition hai. Agar model ka confidence score 70 se kam hai to system user ko known maanega. Ye value project demo ke liye practical starting point hai; real project me dataset aur lighting ke according tune karni padti hai.

`labels.get(user_id, "Unknown")` predicted ID ka name nikalta hai. Agar ID labels dictionary me nahi hai to default `Unknown` use hota hai.

`display_text` screen par show hone wala name/confidence text hai. `box_color` known user ke liye green aur unknown ke liye red set karta hai.

```python
if user_id not in marked_users:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance_row = pd.DataFrame(
        [[user_id, user_name, current_time]],
        columns=["User ID", "Name", "Time"]
    )
```

`if user_id not in marked_users:` duplicate attendance prevent karta hai. Same user ek run me sirf ek baar mark hoga.

`datetime.now().strftime()` current date-time ko readable format me convert karta hai. Attendance record me exact time important hota hai.

`pd.DataFrame(...)` attendance row ko table format me banata hai. Inner list `[[user_id, user_name, current_time]]` ek row represent karti hai. `columns` readable headings define karta hai.

```python
attendance_row.to_csv(
    attendance_file,
    mode="a",
    header=not attendance_file.exists(),
    index=False
)
marked_users.add(user_id)
```

`to_csv()` attendance row ko CSV file me save karta hai. `mode="a"` append mode hai, yani purana data delete nahi hoga. `header=not attendance_file.exists()` first row par header add karta hai, baad me header repeat nahi karta. `index=False` extra index column save hone se rokta hai.

`marked_users.add(user_id)` recognized user ko marked set me add karta hai. Iske baad same session me same user ki duplicate entry nahi hogi.

```python
result_path = RESULTS_DIR / f"{user_name}_{datetime.now().strftime('%H%M%S')}.jpg"
cv2.imwrite(str(result_path), frame)
```

`result_path` screenshot ka file path banata hai. File name me user name aur current time include hota hai, isliye har result image unique hoti hai.

`cv2.imwrite(str(result_path), frame)` current webcam frame ko image file me save karta hai. Ye recognition proof ke form me use ho sakta hai.

```python
cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
cv2.imshow("Face Recognition Attendance", frame)
```

`cv2.rectangle()` face ke around box draw karta hai. `box_color` green ya red ho sakta hai. `cv2.putText()` face ke upar user name ya Unknown show karta hai. `cv2.imshow()` final live output screen par show karta hai.

```python
if cv2.waitKey(1) & 0xFF == ord("q"):
    break

camera.release()
cv2.destroyAllWindows()
```

`cv2.waitKey(1)` keyboard input check karta hai. `ord("q")` q key ka code hota hai. Agar q press hota hai to loop stop ho jata hai.

`camera.release()` webcam free karta hai. `cv2.destroyAllWindows()` OpenCV windows close karta hai. Ye dono cleanup lines project ko properly finish karti hain.

---

# Common Errors and Simple Fixes

`ModuleNotFoundError: No module named 'cv2'` ka matlab OpenCV install nahi hai. Fix: `pip install -r requirements.txt` run karo.

`cv2.face module not found` ka matlab `opencv-contrib-python` missing hai. Is project me LBPH face recognizer ke liye contrib package required hai. Fix: `pip install opencv-contrib-python`.

`Camera could not be opened` ka matlab webcam permission off hai, camera kisi aur app me busy hai, ya camera index wrong hai. Mac me System Settings me camera permission allow karo. External camera use ho raha hai to `cv2.VideoCapture(0)` ko `cv2.VideoCapture(1)` try kar sakte ho.

`No training images found` ka matlab capture step complete nahi hua. Fix: pehle `python3 src/01_capture_images.py` run karo, user images capture karo, phir training script run karo.

`Trained model or labels file not found` ka matlab model training complete nahi hui. Fix: `python3 src/02_train_model.py` run karo.

Recognition baar-baar `Unknown` show kar raha hai to lighting improve karo, face camera ke saamne straight rakho, har user ke 50 clear images capture karo, aur training script dobara run karo. Threshold `70` ko classroom experiment ke liye thoda adjust bhi kiya ja sakta hai.
