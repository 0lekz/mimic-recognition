# Mimic Recognition

https://www.youtube.com/shorts/j7u5kU45kwE

A lightweight real-time facial emotion recognition system built for demo purposes. It detects faces in webcam frames, runs inference with a custom fine-tuned model, and triggers UI pop-ups + audio feedback for each detected emotion.

Developed CNN architecture, originally trained on FER2013, then fine-tuned on a custom dataset (my own web cam data) to improve robustness.

## Features

- **Custom CNN Model**
  - Implemented from scratch (no pretrained models)
  - 3 convolutional layers + small classifier head
  - Trained on 48×48 grayscale faces

- **Own Dataset Creation**
  -	Custom dataset collected manually
  -	Explicit class_to_idx mapping

- **Fine-Tuning**
  - Freeze conv1 & conv2
  - Unfreeze conv3
  - Retrain classifier head

- **Real-Time Demo**
  - Webcam face detection (OpenCV cascade)
  - On-the-fly predictions (Torch)
  - Overlay of live probabilities for all classes
  - Popup window “reaction images”
  - Non-blocking sound playback
  - Logging to CSV with timestamps + confidence scores
 
<img width="788" height="539" alt="demoscreenshot" src="https://github.com/user-attachments/assets/06235878-ba6c-450f-8a75-6c6794f14e16"/>

...
