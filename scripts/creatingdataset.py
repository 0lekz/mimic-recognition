## Creating Dataset from my personal webcam to finetune model
"""_summary_
each iteration of the loop is one frame from the webcam
- `faces` detects all faces in that frame (will be just mine)
- The loop crops and resizes the face, draws a rectangle for visualization
- Then check key presses (cv2.waitKey(1)):
        - If pressed 'h', save the cropped face to the happy folder
        - If pressed 's', save to sad
        - If pressed 'n', save to neutral
        - If pressed 'q', exit the loop and stop capturing
"""

import cv2
import os
import time

# -------
# Configuration
# -------
output_dir = "dataset"
emotion_labels = ["happy", "sad", "neutral"]
frame_skip = 5  # Capture every 5th frame
capture_duration = 60  # Capture for 60 seconds
resize_dim = (48, 48)  # Resize frames to 48x48

# Create folders for each emotion
for emo in emotion_labels:
    emo_dir = os.path.join(output_dir, emo)
    os.makedirs(emo_dir, exist_ok=True)

# Load face detectro from OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'h' for happy, 's' for sad, 'n' for neutral, 'q' to quit.")
start_time = time.time()

# main loop
frame_count = 0
save_count = {emo: 0 for emo in emotion_labels}  # to track saved images per emotion
run_prefix = input("Enter a prefix for this recording session (e.g., 'run1'): ").strip()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)  # flip horizontally (1 means horizontal)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        face_img = gray[y : y + h, x : x + w]
        face_resized = cv2.resize(face_img, resize_dim)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # SAVE FRAME ON KEY PRESS
        key = cv2.waitKey(1) & 0xFF
        if key == ord("h"):
            path = os.path.join(
                output_dir, "happy", f"img_{run_prefix}_{save_count['happy']:04d}.jpg"
            )
            cv2.imwrite(path, face_resized)
            save_count["happy"] += 1
            print(f"[HAPPY] saved {path}")
        elif key == ord("n"):
            path = os.path.join(
                output_dir,
                "neutral",
                f"img_{run_prefix}_{save_count['neutral']:04d}.jpg",
            )
            cv2.imwrite(path, face_resized)
            save_count["neutral"] += 1
            print(f"[NEUTRAL] saved {path}")
        elif key == ord("s"):
            path = os.path.join(
                output_dir, "sad", f"img_{run_prefix}_{save_count['sad']:04d}.jpg"
            )
            cv2.imwrite(path, face_resized)
            save_count["sad"] += 1
            print(f"[SAD] saved {path}")
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            print("Exiting...")
            exit()

        cv2.imshow("Webcam - Press h/s/n to save, q to quit", frame)
        frame_count += 1

# exit
cap.release()
cv2.destroyAllWindows()
