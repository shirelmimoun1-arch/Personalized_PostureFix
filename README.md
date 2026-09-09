# PostureFix AI – Personalized Real-Time Posture Monitoring

![PostureFix AI Cover](media/posturefixai_cover_realistic.png)

## Project Description

PostureFix AI is a personalized real-time posture monitoring system that uses a standard webcam to detect poor sitting posture and provide interpretable corrective feedback.

The system is designed to account for individual differences in natural sitting posture. Instead of relying on a single posture model for all users, each user performs a short calibration session that provides the user-specific reference samples used by a personalized K-Nearest Neighbors (KNN) classifier.

The complete pipeline includes:

- guided personalized posture calibration,
- guided frame labeling and posture segmentation,
- MediaPipe-based pose estimation,
- geometric feature extraction,
- shared feature selection across personalized users,
- personalized KNN training and evaluation,
- real-time posture classification,
- temporal stabilization of posture predictions,
- model-aware personalized corrective feedback.

The final personalized classifier uses a compact seven-feature representation selected through backward sequential feature elimination.

---

## System Demonstration

A video demonstration of the complete PostureFix AI system, including personalized
calibration, real-time posture monitoring, and corrective feedback, is available here:

[▶ Watch the PostureFix AI Demo](https://drive.google.com/file/d/1ODnbpZYtLPPaPkcnGPIszOWj7yEXfLGM/view)

## System Pipeline

The personalized pipeline consists of the following stages:

1. **Calibration**

   The user performs guided examples of Good and Bad sitting posture.

   A natural reading stage is also recorded to capture realistic movement while maintaining correct posture.

   Calibration frames are automatically labeled and grouped into posture segments.

2. **Dataset Construction**

   MediaPipe Pose is applied to each valid calibration frame.

   Pose landmarks are converted into interpretable geometric posture features using the shared feature-extraction module.

3. **Feature Selection**

   A shared feature representation is selected across personalized users using backward sequential feature elimination.

   Feature subsets are evaluated using segment-grouped cross-validation, while the held-out test segments remain excluded from the feature-selection process.

4. **Personalized Model Training**

   A separate KNN classifier is trained for each user.

   Hyperparameters are selected using `StratifiedGroupKFold`, where `Segment_ID` is used as the grouping variable.

   Median imputation and feature standardization are fitted independently inside each training fold.

5. **Held-Out Evaluation**

   Complete posture segments are reserved for final evaluation.

   This prevents temporally related frames from the same recorded posture segment from appearing in both model development and final evaluation.

6. **Final Model Refit**

   After evaluation is complete, the selected preprocessing pipeline and KNN configuration are refitted using the complete calibration dataset.

   This final model is saved for real-time inference.

7. **Live Monitoring**

   Webcam frames are processed continuously using the same MediaPipe and feature-extraction pipeline used during training.

   Frame-level KNN predictions are stabilized using time-based posture-state logic.

8. **Personalized Feedback**

   For persistent Bad posture, the system analyzes the local geometry of the personalized KNN model to determine which posture characteristics provide the strongest evidence for the current prediction.

   Supplementary personalized geometric diagnostics, such as shoulder asymmetry, may also provide corrective guidance without modifying or explaining the KNN classification itself.

---

## Final Personalized Features

The final KNN classifier uses the following seven features:

- `earCenterHeightNorm`
- `eyeCenterHeightNorm`
- `eyeWidth`
- `headForwardDepth`
- `headHeight`
- `leftEar_y`
- `rightEye_y`

The shared seven-feature subset was selected using backward sequential feature elimination across 11 personalized users. 
For each user, feature selection was performed exclusively on the Train/CV data, while the held-out test segments remained excluded from the selection process.

---

## Project Structure

```text
.
├── assets/
│   └── calibration_images/
│       └── Reference posture images used during guided calibration
│
├── calibration/
│   ├── frame_extraction.py
│   │   └── Guided calibration and automatic frame labeling
│   │
│   └── create_user_dataset.py
│       └── Creation of the personalized feature dataset
│
├── features/
│   ├── feature_extraction.py
│   │   └── Shared geometric feature extraction for training and inference
│   │
│   └── posture_feedback_model_aware.py
│       └── Personalized model-aware corrective feedback
│
├── live/
│   └── live_monitoring.py
│       └── Real-time personalized posture inference and feedback
│
├── training/
│   ├── train_personal_model.py
│   │   └── Personalized KNN training, validation, and evaluation
│   │
│   └── feature_selection_experiment.py
│       └── Shared backward sequential feature selection
│
├── media/
│   └── posturefixai_cover_realistic.png
│       └── Project cover image
│
├── main.py
│   └── End-to-end interactive application pipeline
│
├── .gitignore
└── README.md
```

---

## Technologies

The project is implemented in Python and uses:

- OpenCV
- MediaPipe Pose
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Joblib

The machine-learning component uses a personalized K-Nearest Neighbors classifier together with median imputation and feature standardization.

---

## Installation

Recommended environment: **Python 3.11**.

Clone the repository:

```bash
git clone git@github.cs.huji.ac.il:shirlel25/Personalized-PostureFix-AI.git
cd Personalized-PostureFix-AI
```

Install the required Python packages:

```bash
pip install numpy pandas opencv-python mediapipe scikit-learn matplotlib joblib openpyxl
```

A webcam is required for calibration and live monitoring.

---

## Running the Project

Run the application from the repository root:

```bash
python main.py
```

The application asks for a user name or ID and then presents the available actions.

### Available Workflows

#### 1. New calibration

Runs the complete personalized pipeline:

```text
Calibration
    ↓
Dataset construction
    ↓
Personalized model training
    ↓
Held-out evaluation
    ↓
Final model refit
    ↓
Live monitoring
```

#### 2. Recreate the dataset and retrain

Previously recorded calibration frames and labels are reused to regenerate the feature dataset and retrain the personalized model.

#### 3. Retrain from an existing dataset

The existing calibration feature dataset is used directly for personalized model training.

#### 4. Live monitoring

An existing personalized model is loaded and used for real-time posture monitoring.

---

## Personalized Calibration

During a new calibration session, the system records three types of posture behavior:

### Good Posture

The user follows guided reference images demonstrating correct sitting posture.

### Natural Good-Posture Reading

The user reads text while maintaining comfortable Good posture.

This stage introduces natural head and upper-body movement into the Good calibration data.

### Bad Posture

The user follows guided examples of poor sitting posture.

Each guided posture is stored as a separate `Segment_ID`.

The reading stage is also divided into temporal subsegments.

This segment structure is later used to prevent frames belonging to the same recorded posture instance from being divided between training and evaluation.

---

## Feature Extraction

MediaPipe Pose landmarks are extracted from each calibration and live webcam frame.

The shared feature-extraction module computes interpretable geometric descriptors including:

- head position,
- head height,
- forward-head displacement,
- eye and ear geometry,
- shoulder alignment,
- relative head and shoulder orientation,
- selected landmark coordinates and normalized geometric descriptors.

Several measurements are normalized by shoulder width to reduce sensitivity to body size and user-camera distance.

The same feature-extraction implementation is used during both dataset creation and live inference to maintain consistency between training and deployment.

---

## Model Training and Evaluation

For each personalized user, complete posture segments are first divided into:

```text
Train/CV segments
Held-out test segments
```

The held-out segments are not used for model selection.

KNN hyperparameters are selected exclusively from the Train/CV pool using `StratifiedGroupKFold`.

The hyperparameter search evaluates:

- K ∈ {3, 5, 7, 9, 11, 15, 21, 31, 41, 51}
- weights ∈ {uniform, distance}
- metric ∈ {Euclidean, Manhattan}

Within each cross-validation fold:

```text
Training fold
    ↓
Median imputation
    ↓
Standardization
    ↓
KNN fitting
    ↓
Validation fold evaluation
```

Hyperparameter configurations are ranked primarily by Balanced Accuracy, followed by Bad-class F1 and cross-validation stability.

The selected configuration is then evaluated once on the held-out posture segments.

Reported evaluation metrics include:

- Accuracy
- Balanced Accuracy
- Bad-class Recall
- Bad-class F1
- Classification Report
- Confusion Matrix

After evaluation, the final deployment model is refitted using all available calibration samples.

---

## Feature Selection

The feature extractor produces 55 candidate model-input features. Before backward elimination, 
four projection-based features are removed because their geometric definitions make them structurally 
redundant with each other and with `shoulderTilt`, leaving 51 candidate features.

Backward sequential feature elimination was then performed jointly across the 11 personalized users, 
reducing the shared representation from 51 candidate features to the final seven-feature subset.

For every candidate feature subset:

- each user is evaluated independently,
- posture segments remain grouped during cross-validation,
- preprocessing is fitted only on the corresponding training fold,
- KNN hyperparameters are re-optimized,
- user-level results are aggregated with equal weighting across users.

The held-out test segments are excluded from the entire feature-selection process.

The final representation is selected from the evaluated backward-elimination trajectory according to:

1. Mean Balanced Accuracy
2. Mean Bad-class F1
3. Between-user performance variability
4. Number of features as a final tie-breaker

---

## Real-Time Monitoring

During live monitoring:

1. A mirrored webcam frame is acquired.
2. MediaPipe Pose landmarks are detected.
3. The shared geometric features are extracted.
4. The fitted imputer and scaler are applied.
5. The personalized KNN predicts Good or Bad posture.
6. A time-based smoothing mechanism stabilizes the raw prediction.
7. Persistent Bad posture activates personalized corrective feedback.
8. Feedback messages are temporally stabilized before being displayed.

Using elapsed time rather than a fixed number of frames makes the state logic less dependent on the effective processing frame rate.

---

## Personalized Feedback

The primary corrective-feedback mechanism is model-aware.

For a frame classified as Bad, the system:

1. reproduces the preprocessing used by the personalized classifier,
2. examines nearby Good and Bad calibration samples in the KNN feature space,
3. compares their feature-wise distances to the current posture,
4. estimates local feature evidence,
5. groups related features into interpretable posture categories,
6. generates a personalized corrective message.

The current model-aware groups include:

- forward-head posture,
- vertical head posture,
- head orientation.

The system may additionally evaluate selected geometric quantities that are not part of the classifier.

For example, `shoulderTilt` is compared with the user's calibrated Good-posture reference to detect unusual shoulder asymmetry.

Such supplementary diagnostics do **not** participate in the KNN decision and are not interpreted as explanations of the classifier prediction.

---

## Evaluation Summary

The final personalized evaluation was performed on held-out posture segments from 11 users, comprising 2,496 test frames. 
The personalized pipeline achieved 87.10% pooled accuracy and 86.88% weighted F1-score.

The earlier generalized model achieved 78.87% accuracy on a participant-independent test set of 2,471 frames from unseen users. 
Because the generalized and personalized systems were evaluated under different protocols, these results should not be interpreted 
as a controlled head-to-head comparison.

## Output Files

For each user, the application creates a user-specific directory containing calibration and evaluation artifacts.

Example:

```text
users/
└── <user_id>/
    ├── frames/
    ├── labels.csv
    ├── calibration_dataset.xlsx
    ├── landmark_checks/
    └── evaluation/
```

Personalized trained models are stored under:

```text
models/
└── user_models/
    └── <user_id>_personal_knn.joblib
```

---

## Authors

**Shirel Maimon**  
**Ester Fradkin**

The Hebrew University of Jerusalem

---

## Acknowledgments

This project was developed as an academic project at the Hebrew University of Jerusalem.

We would like to thank our advisor and mentor, Gal Katzhendler, for his guidance and support throughout the project, and Nir Sweed for his valuable assistance and contributions.

The implementation uses MediaPipe Pose for body-landmark estimation and scikit-learn for personalized KNN modeling and evaluation.
