# PostureFix AI – Personalized Posture Monitoring

## Overview

**PostureFix AI** is a personalized posture monitoring system that uses computer vision and machine learning to detect whether a user is sitting in a healthy or unhealthy posture.

Unlike generic posture classifiers, this project trains a **personalized model** for each user. During a short calibration session, the user records examples of both good and bad postures. The system automatically labels the captured images, extracts posture features using **MediaPipe Pose**, and trains a personalized **K-Nearest Neighbors (KNN)** classifier.

The trained model can later be used to recognize the user's posture in real time and provide personalized feedback.

---

## Features

* Webcam-based personalized calibration
* Guided collection of good and bad posture examples
* Automatic frame extraction and labeling
* MediaPipe Pose landmark detection
* Mathematical posture feature extraction
* Personalized KNN classifier
* Segment-based train/test splitting
* GroupKFold cross-validation for hyperparameter selection
* Automatic evaluation reports and confusion matrix
* Misclassified sample analysis

---

## Pipeline

```
Webcam Calibration
        │
        ▼
Frame Extraction
        │
        ▼
Automatic Labeling
        │
        ▼
MediaPipe Pose Detection
        │
        ▼
Feature Extraction
        │
        ▼
Personalized Dataset
        │
        ▼
KNN Training
        │
        ▼
Personalized Posture Prediction
```

---

## Repository Structure

```
.
├── frame_extraction.py
├── create_user_dataset.py
├── train_personal_model.py
├── features/
│   └── feature_extraction.py
├── assets/
│   └── calibration_images/
│       ├── good/
│       └── bad/
├── data/
├── models/
└── evaluation/
```

---

## Calibration

The calibration process guides the user through a series of predefined postures.

The user is shown reference images and asked to imitate them while the webcam records frames.

The calibration consists of:

* Multiple **good posture** examples
* A **natural reading session** while maintaining good posture
* Multiple **bad posture** examples, including:

  * Head down
  * Leaning forward
  * Leaning left/right
  * Head tilted
  * Shoulder asymmetry
  * Rounded shoulders
  * Upper body rotation
  * Supporting the head with one hand

Each captured frame is automatically labeled and assigned a segment identifier for later machine learning.

---

## Dataset Creation

After calibration, each frame is processed using **MediaPipe Pose**.

For every image the system:

1. Detects body landmarks.
2. Computes mathematical posture features.
3. Saves the extracted features together with:

   * Image ID
   * Label
   * Segment ID

The resulting dataset is saved as:

```
calibration_dataset.xlsx
```

---

## Model Training

The personalized classifier is based on **K-Nearest Neighbors (KNN).**

Before training:

* Missing values are imputed using the median.
* Features are standardized using `StandardScaler`.

The training pipeline performs:

* Segment-based train/test split
* GroupKFold cross-validation
* Automatic hyperparameter search
* Final model training on the complete dataset

The trained model is saved using Joblib.

---

## Evaluation

The project automatically generates:

* Classification report
* Confusion matrix
* Accuracy
* Hyperparameter search results
* GroupKFold validation results
* Misclassified samples
* Images that were incorrectly classified

Example evaluation directory:

```
evaluation/
├── classification_report.txt
├── confusion_matrix.png
├── metrics_summary.csv
├── split_summary.txt
├── hyperparameter_selection/
│   ├── group_cv_search_results.csv
│   └── K_vs_group_cv_accuracy.png
└── misclassified/
    └── bad_predicted_good/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/PostureFix-AI.git
cd PostureFix-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install opencv-python mediapipe pandas numpy scikit-learn matplotlib joblib openpyxl
```

---

## Usage

### Step 1 — Personalized Calibration

Run the calibration process:

```python
from frame_extraction import run_calibration

run_calibration(
    frames_dir="data/frames",
    labels_csv_path="data/labels.csv",
    target_fps=5,
    good_reference_dir="assets/calibration_images/good",
    bad_reference_dir="assets/calibration_images/bad"
)
```

---

### Step 2 — Create the Dataset

```python
from create_user_dataset import create_user_dataset

create_user_dataset(
    frames_dir="data/frames",
    labels_csv_path="data/labels.csv",
    output_file="data/calibration_dataset.xlsx"
)
```

---

### Step 3 — Train the Personalized Model

```python
from train_personal_model import train_personal_model

train_personal_model(
    dataset_path="data/calibration_dataset.xlsx",
    model_output_path="models/personal_model.joblib",
    frames_dir="data/frames",
    evaluation_dir="evaluation"
)
```

---

## Technologies

* Python
* OpenCV
* MediaPipe Pose
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Joblib

---

## Future Work

Future improvements include:

* Fatigue detection
* Personalized ergonomic recommendations
* Break reminders
* Long-term posture statistics

---

## Motivation

Poor sitting posture is one of the most common causes of neck, shoulder, and back pain among people who spend long hours working at a computer.

Most existing posture monitoring systems use generic models that may not accurately reflect each individual's natural posture.

This project introduces a **personalized machine learning approach**, allowing every user to build a posture classifier tailored specifically to their own body, sitting habits, and camera setup.

---

## Authors

**Shirel Mimoun**

**Ester Fradkin**

Hebrew University of Jerusalem

Final Project – Personalized AI Posture Monitoring
