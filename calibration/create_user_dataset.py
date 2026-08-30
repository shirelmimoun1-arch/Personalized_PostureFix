# ============================================================
# CREATE USER DATASET – PERSONALIZED CALIBRATION
# ============================================================
#
# Purpose:
#   Create a machine-learning dataset from the user's automatically
#   labeled calibration frames.
#
# Input:
#   - frames_dir:
#       Folder containing calibration frame images.
#
#   - labels_csv_path:
#       CSV file created during automatic calibration.
#       Contains: Image_ID, Label, Segment_ID
#
# Output:
#   - calibration_dataset.xlsx:
#       One row per frame, containing:
#       Image_ID, Label, Segment_ID, extracted posture features,
#       and raw landmark coordinates.
#
# Notes:
#   - Labels and segment IDs come from labels.csv, not from CVAT.
#   - Segment_ID is used later for segment-based train/test splitting.
#   - Unlabeled frames are skipped.
#   - MediaPipe is initialized only inside create_user_dataset()
#     so importing this file does not start MediaPipe immediately.
# ============================================================

import cv2
import mediapipe as mp
import pandas as pd
import os

from features.feature_extraction import extract_math_features


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ============================================================
# LABEL AND SEGMENT LOADING
# ============================================================

def load_labels(labels_csv_path):
    """
    Load automatically generated labels from labels.csv.

    Returns:
        Dictionary mapping:
            Image_ID -> {
                "Label": label,
                "Segment_ID": segment_id
            }
    """

    if not os.path.exists(labels_csv_path):
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv_path}")

    labels_df = pd.read_csv(labels_csv_path)

    required_columns = ["Image_ID", "Label", "Segment_ID"]
    for col in required_columns:
        if col not in labels_df.columns:
            raise RuntimeError(
                f"Missing required column '{col}' in labels.csv"
            )

    label_info = {}

    for _, row in labels_df.iterrows():
        label_info[row["Image_ID"]] = {
            "Label": row["Label"],
            "Segment_ID": row["Segment_ID"]
        }

    return label_info


# ============================================================
# DATASET CREATION
# ============================================================

def create_user_dataset(frames_dir, labels_csv_path, output_file):
    """
    Create the personalized calibration dataset.

    For each saved frame:
        1. Read image.
        2. Check that it has label information in labels.csv.
        3. Copy Label and Segment_ID into the dataset row.
        4. Run MediaPipe Pose.
        5. Extract posture features.
        6. Save one row to calibration_dataset.xlsx.
    """

    landmark_dir = os.path.join(os.path.dirname(output_file), "landmark_checks")
    os.makedirs(landmark_dir, exist_ok=True)

    labels = load_labels(labels_csv_path)
    dataset = []

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
    ) as pose:

        for filename in os.listdir(frames_dir):

            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            if filename not in labels:
                continue

            img_path = os.path.join(frames_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            row = {
                "Image_ID": filename,
                "Label": labels[filename]["Label"],
                "Segment_ID": labels[filename]["Segment_ID"]
            }

            result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # if there is no pose, skip frame
            if not result.pose_landmarks:
                continue

            features = extract_math_features(result.pose_landmarks.landmark)
            row.update(features)

            annotated = img.copy()

            mp_drawing.draw_landmarks(
                annotated,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )

            cv2.imwrite(
                os.path.join(landmark_dir, f"check_{filename}"),
                annotated
            )

            dataset.append(row)

    df = pd.DataFrame(dataset)
    df.to_excel(output_file, index=False)

    print(f"\nDataset saved to: {output_file}")
    print(f"Landmark check images saved to: {landmark_dir}")