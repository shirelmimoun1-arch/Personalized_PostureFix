# PostureFix AI – Personalized Posture Monitoring

PostureFix AI is a personalized posture-monitoring system that uses a webcam and machine learning to detect whether a user is sitting in a healthy or unhealthy posture.

The system is designed around personal calibration: each user records examples of their own good and bad sitting postures, the system extracts pose-based features using MediaPipe, and then trains a personalized KNN classifier.

## Features

- Webcam-based posture calibration
- Guided recording of good and bad posture examples
- Automatic frame extraction and labeling
- MediaPipe Pose landmark detection
- Mathematical posture feature extraction
- Personalized KNN model training
- Segment-based train/test splitting to avoid frame leakage
- GroupKFold cross-validation for hyperparameter selection
- Evaluation reports, confusion matrix, and misclassification analysis

## Project Pipeline

```text
Calibration Video / Webcam
        |
        v
Frame Extraction + Automatic Labels
        |
        v
MediaPipe Pose Landmark Detection
        |
        v
Posture Feature Extraction
        |
        v
Personalized Dataset Creation
        |
        v
KNN Model Training
        |
        v
Good / Bad Posture Prediction
