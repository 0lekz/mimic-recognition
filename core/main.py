import cv2
import torch
import torch.nn.functional as F
import time
import csv
from datetime import datetime
from pathlib import Path
from playsound3 import playsound
from model import MyCNN

# ---
# Load the model
# ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = MyCNN().to(device)
model.load_state_dict(
    torch.load("models/fine_tuned_classifier12.pth", map_location=device)
)
model.eval()

# ---
# Face detector
# ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class_names = ["happy", "sad", "neutral"]

# Log
log_path = Path("predictions_log.csv")
if not log_path.exists():
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp", "emotion", "confidence"])

cap = cv2.VideoCapture(0)

pause_until = 0  # pauses prediction only
popup_until = 0  # controls popup timer
popup_img = None  # image shown in popup

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_time = time.time()

    # --- detect faces always ---
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # --- draw rectangles always ---
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- show popup if active ---
    if current_time < popup_until and popup_img is not None:
        cv2.imshow("bruh", popup_img)
    else:
        if popup_img is not None:
            cv2.destroyWindow("bruh")
            popup_img = None

    cv2.imshow("FER Live", frame)
    if current_time < pause_until:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # --- do prediction ---
    for x, y, w, h in faces:
        face = gray[y : y + h, x : x + w]
        face_resized = cv2.resize(face, (48, 48))

        face_tensor = (
            torch.tensor(face_resized).unsqueeze(0).unsqueeze(0).float() / 255.0
        )
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.to(device)

        with torch.no_grad():
            output = model(face_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            emotion = class_names[pred.item()]
            confidence = conf.item()

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [datetime.now().isoformat(), emotion, round(confidence, 4)]
            )

        # Show prob bar
        prob_text = " | ".join(
            f"{cls}: {probs[0, i].item() * 100:.1f}%"
            for i, cls in enumerate(class_names)
        )
        cv2.putText(
            frame,
            prob_text,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Label
        cv2.putText(
            frame,
            f"{emotion} ({confidence * 100:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # --- popup logic ---
        if emotion == "sad":
            popup_img = cv2.imread("../assets/sad.jpg")
            playsound("../assets/sad_short.mp3", block=False)
            print("Sad detected!")
            popup_until = time.time() + 1.5
            pause_until = time.time() + 1.5

        elif emotion == "happy":
            popup_img = cv2.imread("../assets/happy.jpg")
            playsound("../assets/boom.mp3", block=False)
            print("Happy detected!")
            popup_until = time.time() + 1.5
            pause_until = time.time() + 1.5

    cv2.imshow("FER Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
