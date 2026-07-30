# Face Recognition System Project

# Complete Step-by-Step Classroom Notes in Simple Hinglish

## Project Overview

Is project me hum Python, OpenCV aur basic Machine Learning ka use karke ek **Face Recognition System** build karenge. Ye project step-by-step banega. Hum ek saath pura code nahi likhenge. Pehle har topic ko detail me samjhenge, phir us topic ka practical code likhenge, phir code ki har line ko simple Hinglish me explain karenge.

Is project ka final goal hai webcam se face detect karna, user ke face images capture karna, captured images se face recognition model train karna, registered user ko recognize karna, real-time webcam me user ka naam show karna, attendance mark karna, aur recognition result save karna.

Real-world story: Imagine karo ek classroom, office, coaching center ya lab hai. Wahan attendance manually lena time-consuming hota hai. Agar system automatically face detect karke registered student ya employee ko identify kare aur attendance save kare, to time save hota hai. Ye project isi idea ka beginner-friendly version hai.

Important note: Face recognition real world me privacy-sensitive technology hai. Is project ko sirf learning, demo, classroom aur consent-based use ke liye banana chahiye. Kisi ka face data bina permission capture ya use nahi karna chahiye.

---

# Teacher Read-Aloud Master Notes

## Project Purpose - Ye Project Kyu Bana Rahe Hain?

Students, is project ka purpose sirf face recognition ka code chalana nahi hai. Is project ka actual purpose ye samajhna hai ki real-world computer vision project ka complete flow kaise work karta hai. Jab koi company ya institute face-based attendance system banata hai, to wo direct recognition se start nahi karta. Sabse pehle face detect hota hai, phir registered users ke face images collect hote hain, phir un images se model train hota hai, phir model real-time camera me face ko identify karta hai, phir attendance save hoti hai, aur end me result ka record maintain hota hai.

Is project me students ko ye samajhna hai ki har step ek dusre se connected hai. Agar face detection weak hai, to captured images weak hongi. Agar captured images weak hain, to model training weak hogi. Agar model training weak hai, to recognition wrong ho sakta hai. Isliye project ko step by step build karna important hai. Hum ek saath pura code nahi likhenge, kyunki students ko sirf copy-paste nahi, balki project ka logic samajhna hai.

Real-world me aise systems classroom attendance, office entry, lab access, library entry, exam verification, visitor management aur security monitoring me use ho sakte hain. Lekin students ko ye bhi samjhana zaroori hai ki face recognition privacy-sensitive technology hai. Kisi ka face data bina permission collect nahi karna chahiye. Learning purpose ke liye hamesha consent ke saath use karna chahiye.

## Topic 1 - Face Detection using OpenCV

Face Detection ka matlab hota hai image ya video frame ke andar face ka location find karna. Yahan system ko ye nahi pata hota ki face Rahul ka hai, Priya ka hai, ya kisi aur ka. System sirf ye identify karta hai ki image me face kaha present hai. Face detection recognition se pehle ka step hai. Agar face detect hi nahi hoga, to recognition possible nahi hoga.

OpenCV ek computer vision library hai. Computer vision ka matlab computer ko image aur video samjhana. OpenCV se hum webcam open kar sakte hain, frames read kar sakte hain, image ko grayscale me convert kar sakte hain, face detect kar sakte hain, rectangle draw kar sakte hain, aur screen par live output dikha sakte hain.

Is project me hum Haar Cascade use karenge. Haar Cascade OpenCV ka pre-trained face detector hai. Pre-trained ka matlab ye already face patterns par trained hai. Hume face detector ko manually train nahi karna. Hum sirf us detector ko load karenge aur webcam frames me faces detect karenge.

Students ko yahan ye clear karna hai: detection ka kaam hai face kaha hai ye batana. Recognition ka kaam hai face kis person ka hai ye batana. Dono alag concepts hain.

## Topic 2 - Capture Images from Webcam

Face recognition model ko kisi person ko identify karna sikhane ke liye us person ke face images chahiye. Isliye hum webcam se registered user ke multiple face images capture karenge. Har user ka alag folder banega. Example: `dataset/1_Rahul`, `dataset/2_Priya`.

Multiple images isliye capture karte hain kyunki real life me face hamesha same angle me nahi hota. Kabhi user thoda left dekhta hai, kabhi right, kabhi light kam hoti hai, kabhi expression change hota hai. Agar model ko sirf ek image milegi, to model user ka face robustly learn nahi kar paayega. Isliye beginner project me bhi 30-50 images lena better hota hai.

Is step ka goal hai clean face dataset banana. Dataset ke bina model train nahi hoga. Isliye image capture project ka data collection step hai. Data collection real ML project ka bahut important part hota hai.

## Topic 3 - Face Recognition using Machine Learning

Face Recognition ka matlab detected face ko identify karna. Jab webcam me face detect hota hai, to system us face ko trained model ke paas bhejta hai. Model compare karta hai ki ye face registered users me se kis user ke face pattern se match karta hai.

Is project me hum OpenCV ka LBPH Face Recognizer use karenge. LBPH ka full form Local Binary Patterns Histograms hai. Simple words me, ye face ke texture aur pattern ko numbers me represent karta hai. Phir new face aane par ye trained face patterns se compare karta hai.

Students ko yahan ye samjhana hai ki model face ko human ki tarah nahi dekhta. Model image ko pixels aur patterns ki form me dekhta hai. Machine Learning model data se patterns learn karta hai, aur phir new input par prediction karta hai.

## Topic 4 - Train Face Recognition Model

Training ka matlab model ko examples se learning karwana. Humare paas dataset folder me face images hongi. Har image ke saath ek label hoga, jaise user ID 1, user ID 2. Model images aur labels ko dekhkar learn karega ki kis face pattern ka relation kis user ID se hai.

Training ke baad model file save hoti hai, jaise `trainer/face_model.yml`. Ye file model ki learned knowledge store karti hai. Jab recognition script run hoga, wo model file load karega aur real-time face identify karega.

Students ko ye point samjhao: training ke bina recognition possible nahi hai. Pehle data collect hoga, phir model train hoga, phir recognition hoga.

## Topic 5 - Identify Registered Users

Registered user ka matlab wo user jiska face dataset me capture hua hai aur jiske images se model trained hua hai. Jab registered user camera ke saamne aata hai, model us face ko predict karta hai aur user ID return karta hai. User ID se hum user name find karte hain.

Model confidence score bhi return karta hai. LBPH me lower confidence generally better match hota hai. Is project me hum threshold use karenge, jaise confidence 70 se kam hai to user recognized. Agar confidence high hai to Unknown show hoga. Threshold real project me testing ke according adjust hota hai.

## Topic 6 - Real-time Face Detection

Real-time ka matlab live webcam frames par continuously processing karna. Webcam ek second me many frames deta hai. Hum loop ke andar har frame read karte hain, face detect karte hain, model se predict karte hain, rectangle draw karte hain, aur result screen par show karte hain.

Students ko yahan loop ka concept samjhana hai. `while True` ka matlab program continuous chalta rahega jab tak user `q` press karke stop na kare. Ye live application ka base pattern hai.

## Topic 7 - Attendance Marking Basic

Attendance marking ka matlab recognized user ka record CSV file me save karna. Hum user ID, user name aur current time save karenge. Isse attendance file banegi, jaise `attendance/attendance_2026-07-30.csv`.

Duplicate attendance avoid karne ke liye hum `marked_users` set use karenge. Agar user already marked hai, to same session me dobara attendance row add nahi hogi. Ye basic attendance logic hai.

## Topic 8 - Save Recognition Results

Recognition result save karna useful hai kyunki later proof ke roop me screenshot dekha ja sakta hai. Jab user recognized hota hai, system current frame ko `results/` folder me image ke form me save kar sakta hai.

Isse students ko logging ka concept samajh aata hai. Real applications me sirf output screen par dikhana enough nahi hota. Result ka record save karna bhi important hota hai.

---

# Topics Covered

Is project me hum ye topics cover karenge:

- Face Detection using OpenCV
- Capture Images from Webcam
- Face Recognition using Machine Learning
- Train Face Recognition Model
- Identify Registered Users
- Real-time Face Detection
- Attendance Marking Basic
- Save Recognition Results

---

# Project Folder Structure

Project me folders ka role samajhna important hai:

`src/` folder me Python code files rahengi.  
`dataset/` folder me captured face images save hongi.  
`trainer/` folder me trained face recognition model save hoga.  
`attendance/` folder me attendance CSV files save hongi.  
`results/` folder me recognized user ke screenshots save honge.

Ye structure project ko organized rakhta hai. Agar sab kuch ek hi folder me daal denge to project messy ho jaayega. Real projects me folder organization bahut important hota hai.

---

# Step 1 - Face Detection using OpenCV

## Topic Explanation

Face Detection ka matlab image ya webcam frame me face ka location find karna. Is step me system ye nahi batata ki face kis person ka hai. Ye sirf ye batata hai ki image me face kaha hai. Face detection recognition se pehle aata hai.

OpenCV ek popular computer vision library hai. Computer vision ka matlab computer ko images aur videos samjhana. OpenCV ka use webcam access karne, image read karne, face detect karne, rectangle draw karne aur video window show karne ke liye hota hai.

Hum Haar Cascade face detector use karenge. Haar Cascade OpenCV ka pre-trained face detection model hai. Pre-trained ka matlab ye already face detect karna seekh chuka hai. Hume isko from scratch train nahi karna padta.

## Is Step Me Hum Kya Build Kar Rahe Hain?

Is step me hum webcam start karenge, frame read karenge, frame ko grayscale me convert karenge, face detect karenge, aur detected face ke around rectangle draw karenge. Ye same logic image capture aur recognition dono me use hoga.

## Practical Code

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

## Har Line Detail Explanation

`import cv2` OpenCV library import karta hai. OpenCV ka short module name `cv2` hota hai. Hum webcam access, face detection, image conversion aur rectangle draw karne ke liye OpenCV use kar rahe hain.

`CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"` Haar Cascade face detection XML file ka path banata hai. OpenCV ke andar pre-trained cascade files available hoti hain. Ye file frontal face detect karne ke liye use hoti hai.

`face_detector = cv2.CascadeClassifier(CASCADE_PATH)` Haar Cascade file ko load karke face detector object banata hai. Ye object later frames me faces detect karega.

`camera = cv2.VideoCapture(0)` laptop ya computer ke default webcam ko start karta hai. `0` ka matlab default camera. Agar external camera use ho to kabhi-kabhi `1` use karna pad sakta hai.

`while True:` infinite loop start karta hai. Webcam video continuous frames ka stream hota hai, isliye hume baar-baar frame read karna hota hai.

`success, frame = camera.read()` webcam se ek frame read karta hai. `success` batata hai frame successfully mila ya nahi. `frame` actual image hoti hai.

`if not success:` check karta hai agar frame receive nahi hua. Agar camera off hai ya permission issue hai to success false ho sakta hai.

`print("Camera frame not received.")` error message print karta hai taaki user ko problem samajh aaye.

`break` loop stop karta hai. Agar camera se frame hi nahi mil raha, to aage code chalana useful nahi hai.

`gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` color frame ko grayscale me convert karta hai. Haar Cascade generally grayscale image par faster aur better kaam karta hai.

`faces = face_detector.detectMultiScale(...)` grayscale frame me faces detect karta hai. Output me face locations milti hain.

`gray_frame` detector ko input image deta hai.

`scaleFactor=1.2` image ko different scales par check karne ka control karta hai. Face camera se near ya far ho sakta hai, isliye scale check important hota hai.

`minNeighbors=5` detection quality control karta hai. Higher value false detections kam kar sakti hai.

`minSize=(80, 80)` minimum face size set karta hai. Isse bahut chhote noise areas face detect nahi honge.

`for (x, y, w, h) in faces:` detected faces par loop chalata hai. Har face ka x position, y position, width aur height milta hai.

`cv2.rectangle(...)` detected face ke around rectangle draw karta hai. `(0, 255, 0)` green color hai aur `2` line thickness hai.

`cv2.imshow("Face Detection", frame)` webcam frame ko window me show karta hai.

`if cv2.waitKey(1) & 0xFF == ord("q"):` keyboard key check karta hai. Agar user `q` press kare to loop stop hoga.

`camera.release()` webcam resource ko release karta hai. Ye important cleanup step hai.

`cv2.destroyAllWindows()` OpenCV ki windows close karta hai.

---

# Step 2 - Capture Images from Webcam

## Topic Explanation

Face recognition model ko train karne ke liye hume registered users ke face images chahiye. Isliye pehle hum webcam se user ke face images capture karenge. Har user ka separate folder banega, jisme us user ke multiple face images save honge.

Multiple images isliye chahiye kyunki face thoda left, right, up, down, light change, expression change ke saath different dikhta hai. Agar model ko sirf ek image milegi to recognition weak ho sakta hai. Is project me hum beginner level par 50 images capture karenge.

## Is Step Me Hum Kya Build Kar Rahe Hain?

Hum user se numeric ID aur name input lenge. Webcam start hoga. Face detect hoga. Detected face crop hoke grayscale image ke form me `dataset/userid_username/` folder me save hoga.

## Full Script

File: `src/01_capture_images.py`

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

print("Camera started. Look at the camera.")
print("Press q to stop early.")
```

## Har Line Detail Explanation

`import cv2` OpenCV import karta hai. Webcam open karna, face detect karna, image save karna aur video window show karna OpenCV se hoga.

`from pathlib import Path` Path class import karta hai. File/folder path ko clean way me handle karne ke liye use hota hai.

`DATASET_DIR = Path("dataset")` dataset folder ka path store karta hai. Captured face images isi folder ke andar save hongi.

`CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"` Haar Cascade face detector file ka path banata hai.

`DATASET_DIR.mkdir(exist_ok=True)` dataset folder create karta hai. `exist_ok=True` ka matlab agar folder pehle se exist karta hai to error nahi aayega.

`user_id = input(...).strip()` user se numeric ID input leta hai. `strip()` extra spaces remove karta hai. ID model label ke roop me use hogi.

`user_name = input(...).strip().replace(" ", "_")` user ka name input leta hai. Spaces ko underscore me convert karta hai taaki folder name clean rahe.

`user_folder = DATASET_DIR / f"{user_id}_{user_name}"` specific user ke liye folder path banata hai. Example: `dataset/1_Rahul`.

`user_folder.mkdir(exist_ok=True)` user ka folder create karta hai. Isi folder me images save hongi.

`face_detector = cv2.CascadeClassifier(CASCADE_PATH)` face detector object create karta hai.

`camera = cv2.VideoCapture(0)` webcam start karta hai.

`image_count = 0` captured images ka counter start karta hai.

`max_images = 50` maximum 50 images capture karne ka limit set karta hai.

`print(...)` user ko instruction show karta hai ki camera start ho gaya aur `q` press karke stop kar sakte hain.

## Capture Loop Code

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
```

## Har Line Detail Explanation

`while True:` continuous loop start karta hai. Webcam se images continuously aati hain, isliye loop required hai.

`success, frame = camera.read()` webcam se current frame read karta hai.

`if not success:` check karta hai agar frame successfully read nahi hua.

`print("Camera frame not received.")` problem message print karta hai.

`break` loop stop karta hai.

`gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` frame ko grayscale me convert karta hai.

`faces = face_detector.detectMultiScale(...)` grayscale frame me face locations detect karta hai.

`for (x, y, w, h) in faces:` detected faces ke coordinates par loop chalata hai.

`image_count += 1` saved image count ko 1 se increase karta hai.

`face_image = gray_frame[y:y + h, x:x + w]` detected face area crop karta hai. Ye only face image banata hai, full frame nahi.

`image_path = user_folder / f"{image_count}.jpg"` image save karne ka path banata hai.

`cv2.imwrite(str(image_path), face_image)` cropped face image ko JPG file ke form me save karta hai.

---

# Step 3 - Train Face Recognition Model

## Topic Explanation

Ab hum captured face images se model train karenge. Training ka matlab model ko registered users ke faces ke patterns sikhana. Is project me hum OpenCV ka **LBPH Face Recognizer** use karenge.

LBPH ka full form Local Binary Patterns Histograms hai. Simple words me, ye face image ke texture patterns ko learn karta hai. Beginner face recognition projects ke liye LBPH simple aur useful algorithm hai.

## Is Step Me Hum Kya Build Kar Rahe Hain?

Hum `dataset/` folder se images read karenge. Har user folder se user ID nikalenge. Images aur labels ko list me store karenge. Phir recognizer train karenge aur model `trainer/face_model.yml` me save karenge.

## Full Script

File: `src/02_train_model.py`

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

## Har Line Detail Explanation

`import cv2` OpenCV import karta hai. LBPH recognizer OpenCV contrib package me available hota hai.

`import numpy as np` NumPy import karta hai. Labels ko numeric array me convert karne ke liye use hoga.

`from pathlib import Path` folder aur file paths manage karne ke liye import hota hai.

`DATASET_DIR = Path("dataset")` captured images folder ka path store karta hai.

`TRAINER_DIR = Path("trainer")` trained model save karne wale folder ka path store karta hai.

`MODEL_PATH = TRAINER_DIR / "face_model.yml"` trained model file ka path define karta hai.

`LABELS_PATH = TRAINER_DIR / "labels.txt"` user ID aur user name mapping save karne wali file ka path define karta hai.

`TRAINER_DIR.mkdir(exist_ok=True)` trainer folder create karta hai agar exist nahi karta.

`recognizer = cv2.face.LBPHFaceRecognizer_create()` LBPH face recognizer object create karta hai.

`faces = []` face images store karne ke liye empty list banata hai.

`labels = []` user IDs store karne ke liye empty list banata hai.

`label_names = {}` user ID aur user name mapping ke liye dictionary banata hai.

## Training Data Reading Code

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
```

## Har Line Detail Explanation

`for user_folder in DATASET_DIR.iterdir():` dataset folder ke andar available user folders par loop chalata hai.

`if not user_folder.is_dir():` check karta hai ki current item folder hai ya nahi.

`continue` agar item folder nahi hai to usko skip karta hai.

`folder_parts = user_folder.name.split("_", 1)` folder name ko ID aur name me split karta hai. Example: `1_Rahul` se `1` aur `Rahul`.

`user_id = int(folder_parts[0])` folder name ka first part numeric user ID me convert karta hai.

`user_name = ...` folder name se user name nikalta hai. Agar name available nahi hai to default name banata hai.

`label_names[user_id] = user_name` dictionary me ID-name mapping store karta hai.

`for image_path in user_folder.glob("*.jpg"):` user folder ke andar JPG images par loop chalata hai.

`face_image = cv2.imread(..., cv2.IMREAD_GRAYSCALE)` image ko grayscale format me read karta hai.

`if face_image is None:` check karta hai agar image read nahi hui.

`continue` unreadable image ko skip karta hai.

`faces.append(face_image)` face image list me add karta hai.

`labels.append(user_id)` corresponding user ID label list me add karta hai.

## Model Training and Saving Code

```python
if len(faces) == 0:
    print("No training images found. Please run 01_capture_images.py first.")
    raise SystemExit(1)

recognizer.train(faces, np.array(labels))
recognizer.write(str(MODEL_PATH))

with LABELS_PATH.open("w") as file:
    for user_id, user_name in label_names.items():
        file.write(f"{user_id},{user_name}\n")

print("Model training completed.")
```

## Har Line Detail Explanation

`if len(faces) == 0:` check karta hai ki training ke liye images available hain ya nahi.

`print(...)` user ko message deta hai ki pehle images capture karni hongi.

`raise SystemExit(1)` program stop karta hai because training images ke bina model train nahi ho sakta.

`recognizer.train(faces, np.array(labels))` LBPH recognizer ko faces aur labels se train karta hai.

`np.array(labels)` labels list ko NumPy array me convert karta hai, jo recognizer ko required format me chahiye.

`recognizer.write(str(MODEL_PATH))` trained model ko file me save karta hai.

`with LABELS_PATH.open("w") as file:` labels file write mode me open karta hai.

`for user_id, user_name in label_names.items():` dictionary ke ID-name pairs par loop chalata hai.

`file.write(f"{user_id},{user_name}\n")` labels file me user ID aur name save karta hai.

`print("Model training completed.")` training complete message show karta hai.

---

# Step 4 - Real-time Face Recognition

## Topic Explanation

Face Recognition ka matlab detected face ko identify karna. Face Detection sirf face location batata hai. Face Recognition batata hai face kis registered user ka ho sakta hai.

Recognition ke liye hum trained LBPH model load karenge. Webcam se face detect karenge. Cropped face recognizer ko denge. Recognizer user ID aur confidence score return karega. Confidence low ho to match better maana jaata hai.

## Is Step Me Hum Kya Build Kar Rahe Hain?

Hum trained model load karenge, labels read karenge, webcam open karenge, faces detect karenge, model se predict karenge, aur recognized user ka name screen par show karenge.

## Recognition Setup Code

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
```

## Har Line Detail Explanation

`import cv2` OpenCV import karta hai for webcam, face detection and recognition.

`import pandas as pd` attendance CSV create karne ke liye Pandas import karta hai.

`from datetime import datetime` current date and time lene ke liye import hota hai.

`from pathlib import Path` file/folder paths manage karne ke liye import hota hai.

`TRAINER_DIR = Path("trainer")` trained model folder ka path store karta hai.

`ATTENDANCE_DIR = Path("attendance")` attendance files save karne wale folder ka path store karta hai.

`RESULTS_DIR = Path("results")` recognition screenshots save karne wale folder ka path store karta hai.

`MODEL_PATH` trained model file ka path hai.

`LABELS_PATH` user ID-name mapping file ka path hai.

`CASCADE_PATH` Haar Cascade face detector XML file ka path hai.

---

# Step 5 - Attendance Marking Basic

## Topic Explanation

Attendance marking ka matlab recognized user ka ID, name aur time CSV file me save karna. Basic attendance system me agar user recognize ho gaya, to uska attendance row add ho jaata hai.

Is project me same user ko baar-baar attendance mark hone se bachane ke liye `marked_users` set use karenge. Agar user already marked hai, to dobara row add nahi hogi.

## Attendance Code

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

## Har Line Detail Explanation

`attendance_file = ...` current date ke naam se attendance CSV file ka path banata hai.

`datetime.now().date()` aaj ki date return karta hai.

`marked_users = set()` already marked users ko track karne ke liye empty set banata hai.

`if user_id not in marked_users:` check karta hai user ka attendance already marked hai ya nahi.

`current_time = datetime.now().strftime(...)` current date and time formatted string me convert karta hai.

`attendance_row = pd.DataFrame(...)` attendance row ko table format me banata hai.

`[[user_id, user_name, current_time]]` one row ka data hai.

`columns=["User ID", "Name", "Time"]` CSV ke column names define karta hai.

`attendance_row.to_csv(...)` attendance row CSV file me save karta hai.

`mode="a"` append mode hai. Matlab new row existing file ke end me add hogi.

`header=not attendance_file.exists()` agar file first time create ho rahi hai to header add karega, warna header repeat nahi karega.

`index=False` DataFrame ka extra index CSV me save nahi karega.

`marked_users.add(user_id)` user ko marked list me add karta hai taaki same session me duplicate attendance na ho.

---

# Step 6 - Save Recognition Results

## Topic Explanation

Recognition result save karna useful hota hai because later hum proof ya log ke roop me screenshot dekh sakte hain. Jab user recognized hota hai, system current frame ko image file me save kar sakta hai.

## Result Saving Code

```python
result_path = RESULTS_DIR / f"{user_name}_{datetime.now().strftime('%H%M%S')}.jpg"
cv2.imwrite(str(result_path), frame)
```

## Har Line Detail Explanation

`result_path = ...` result image ka file path banata hai. File name me user name aur current time use hota hai.

`datetime.now().strftime('%H%M%S')` current hour, minute and second ko string me convert karta hai.

`cv2.imwrite(str(result_path), frame)` current webcam frame ko image file ke form me save karta hai.

---

# Step 7 - Complete Real-time Recognition Loop

## Topic Explanation

Ab hum final step samjhenge jahan complete real-time recognition hoti hai. Is step me camera continuously frames read karta hai. Har frame grayscale me convert hota hai. Face detect hota hai. Detected face trained model ko diya jaata hai. Model user ID aur confidence return karta hai. Agar confidence threshold ke andar hai, to user recognized maana jaata hai. Agar confidence high hai, to user unknown maana jaata hai.

Confidence ko simple words me distance score samjho. LBPH recognizer me lower confidence usually better match hota hai. Is project me hum `confidence < 70` ko recognized condition maan rahe hain. Real project me threshold testing ke baad adjust karna padta hai.

## Complete Recognition Loop Code

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
```

## Har Line Detail Explanation

`while True:` continuous loop start karta hai. Real-time recognition me camera baar-baar frames deta hai, isliye loop chahiye.

`success, frame = camera.read()` webcam se current frame read karta hai. `success` batata hai frame mila ya nahi. `frame` actual image data hota hai.

`if not success:` check karta hai agar camera frame receive nahi hua. Agar camera permission issue hai ya camera open nahi hua to ye condition true ho sakti hai.

`print("Camera frame not received.")` user ko problem message show karta hai.

`break` loop stop karta hai, kyunki frame ke bina recognition possible nahi.

`gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` color image ko grayscale me convert karta hai. Face detector grayscale image par kaam karta hai.

`faces = face_detector.detectMultiScale(...)` current frame me faces detect karta hai. Ye method face ke positions return karta hai.

`gray_frame` detector ko input image deta hai.

`scaleFactor=1.2` detector ko different sizes ke faces check karne me help karta hai.

`minNeighbors=5` false detection reduce karne me help karta hai. Higher value stricter detection hoti hai.

`minSize=(80, 80)` minimum face size define karta hai. Isse chhote noise areas face nahi maane jaate.

`for (x, y, w, h) in faces:` detected faces ke coordinates par loop chalata hai. Agar ek frame me multiple faces hain, loop multiple times chalega.

`face_image = gray_frame[y:y + h, x:x + w]` detected face area crop karta hai. Recognition model ko full frame nahi, sirf face image deni hoti hai.

`user_id, confidence = recognizer.predict(face_image)` trained model cropped face ko identify karta hai. Model predicted user ID aur confidence score return karta hai.

`if confidence < 70:` confidence threshold check karta hai. Agar score 70 se kam hai to hum face ko recognized maan rahe hain. Ye threshold project ke testing ke according adjust ho sakta hai.

`user_name = labels.get(user_id, "Unknown")` predicted user ID se user name find karta hai. Agar ID labels file me nahi milti to Unknown show karega.

`display_text = f"{user_name} ({round(confidence, 2)})"` screen par show hone wala text banata hai. Isme user name aur confidence score dono show hote hain.

`box_color = (0, 255, 0)` green color set karta hai. Green ka meaning recognized user.

`else:` agar confidence threshold ke andar nahi hai to unknown condition chalegi.

`display_text = "Unknown"` unknown user ke liye display text set karta hai.

`box_color = (0, 0, 255)` red color set karta hai. Red ka meaning unknown face.

`cv2.rectangle(...)` detected face ke around rectangle draw karta hai. Recognized user ke liye green box aur unknown ke liye red box dikhega.

`cv2.putText(...)` rectangle ke upar user name ya Unknown text draw karta hai.

`frame` image hai jisme text draw hoga.

`display_text` wo text hai jo screen par show hoga.

`(x, y - 10)` text ka position hai. Ye face rectangle ke thoda upar text dikhata hai.

`cv2.FONT_HERSHEY_SIMPLEX` text font style define karta hai.

`0.7` font size set karta hai.

`box_color` text ka color same as rectangle rakhta hai.

`2` text thickness define karta hai.

`cv2.imshow("Face Recognition Attendance", frame)` final frame window me show karta hai.

`if cv2.waitKey(1) & 0xFF == ord("q"):` keyboard input check karta hai. Agar user `q` press kare to program stop hoga.

`break` loop stop karta hai.

## Cleanup Code

```python
camera.release()
cv2.destroyAllWindows()

print("Recognition stopped.")
print(f"Attendance saved at: {attendance_file}")
```

## Har Line Detail Explanation

`camera.release()` webcam resource release karta hai. Agar ye line nahi likhenge to camera background me busy reh sakta hai.

`cv2.destroyAllWindows()` OpenCV windows close karta hai.

`print("Recognition stopped.")` user ko message deta hai ki recognition process stop ho gaya.

`print(f"Attendance saved at: {attendance_file}")` attendance CSV file ka path print karta hai, taaki user ko pata chale attendance kaha save hui.

---

# Final Project Flow

Project ko run karne ka flow ye hoga:

1. `python3 src/01_capture_images.py`
2. User ID aur name enter karo.
3. Webcam se 50 face images capture karo.
4. `python3 src/02_train_model.py`
5. Captured images se face model train karo.
6. `python3 src/03_recognize_and_attendance.py`
7. Webcam me registered user ko identify karo.
8. Attendance CSV me save hogi.
9. Recognition screenshot `results/` folder me save hoga.

---

# Final Classroom Summary

Is project me students ne Face Recognition System ka complete practical flow samjha. Pehle face detection samjha, phir webcam se images capture ki, phir captured images se model train kiya, phir trained model se real-time recognition kiya, phir attendance mark ki aur recognition result save kiya.

Important learning ye hai ki Face Recognition project multiple small steps ka combination hai. Agar detection strong nahi hai, capture weak hoga. Agar capture weak hoga, training weak hogi. Agar training weak hogi, recognition weak hoga. Isliye har step ko carefully build karna zaroori hai.
