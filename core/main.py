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
    torch.load("models/fine_tuned_classifier10.pth", map_location=device)
)
model.eval()


# ---
# Load face detector
# ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Emotion labels
class_names = ["happy", "sad", "neutral"]

# Logging setup
log_path = Path

# ---
# main loop
# ---
log_path = Path("predictions_log.csv")
if not log_path.exists():
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "emotion", "confidence"])

cap = cv2.VideoCapture(0)  # 0 for default camera
pause_until = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # flip horizontally (1 means horizontal)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # pause
    current_time = time.time()
    if current_time < pause_until:
        cv2.imshow("FER Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

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

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), emotion, round(confidence, 4)])

        # Draw
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{emotion} ({confidence * 100:.1f}%)"
        cv2.putText(
            frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        if emotion == "sad":
            # CHANGE LATER
            # playsound("alert.mp3", block=False)
            print("Sad detected!")
            pause_until = time.time() + 1.5
        elif emotion == "happy":
            print("Happy detected!")
            # playsound("happy.mp3", block=False)
            pause_until = time.time() + 1.5

    cv2.imshow("FER Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
