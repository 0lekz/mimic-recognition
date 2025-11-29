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

## Project Structure

```bash
mimic-recognition/
├── README.md
├── assets/
│   ├── boom.mp3                 # Sound effects (generic / neutral)
│   ├── boom_short.mp3
│   ├── dataflow.drawio.png      # Dataflow diagram
│   ├── flowofdata.drawio.svg    # Same diagram as SVG
│   ├── happy.jpg                # Reaction image for "happy"
│   ├── sad.jpg                  # Reaction image for "sad"
│   ├── sad.mp3                  # Emotion-specific sounds
│   └── sad_short.mp3
│
├── core/
│   ├── fine_tuning.ipynb                # Fine-tuning on custom webcam dataset
│   ├── main.py                          # Real-time demo entry point (webcam app)
│   ├── mimic-recognition-barebones-fer.ipynb  # Base FER2013 training / barebones setup
│   ├── model.py                         # CNN architecture + model utilities
│   ├── models/                          # Saved model weights / checkpoints
│   ├── predictions_log.csv              # CSV log of real-time predictions
│   └── proof_of_concept.ipynb           # Early POC / experiments
│
├── dataset/
│   ├── 0_happy                          # Class 0 images
│   ├── 1_sad                            # Class 1 images
│   └── 2_neutral                        # Class 2 images
│
├── requirements.txt                     # Python dependencies
│
└── scripts/
    ├── creatingdataset.py               # Script to capture & save webcam frames
    └── merge.py                         # Script to merge/organize datasets
```
## Limitations & Future Work

Current limitations:
- Small custom dataset (only a few emotions, one main subject).
- Sensitive to lighting changes, occlusions, and extreme poses.
- Focused on one-face-at-a-time real-time interaction.

Potential next steps:
- Add more emotion classes and more subjects.
- Replace Haar cascades with a stronger detector (e.g., DNN-based).
- Try deeper models or lightweight pretrained backbones (e.g., MobileNet).
- Wrap the demo in a small GUI app or web interface.
- Better calibration / smoothing (temporal averaging of predictions).

## License

MIT License.

## Contact

For questions or suggestions, open an issue or reach out via GitHub: @0lekz

Mimic Recognition is primarily a learning and demo project. Feel free to fork it, hack on it, and adapt it to your own face, datasets, and ideas.
