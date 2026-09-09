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
#       Image_ID, Label, Segment_ID, and extracted posture features,
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
    Load automatically generated calibration labels.

    Args:
        labels_csv_path (str):
            Path to the labels.csv file containing
            Image_ID, Label, and Segment_ID.

    Returns:
        dict:
            Mapping from Image_ID to its corresponding
            Label and Segment_ID.

    Raises:
        FileNotFoundError:
            If the labels file does not exist.

        RuntimeError:
            If one of the required columns is missing.
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
    Create a personalized posture dataset from calibration frames.

    Each valid frame is processed using MediaPipe Pose, and the
    detected landmarks are converted into posture-related geometric
    features.

    Args:
        frames_dir (str):
            Directory containing the saved calibration frames.

        labels_csv_path (str):
            Path to the calibration labels file.

        output_file (str):
            Path of the Excel file in which the resulting dataset
            will be stored.

    Processing:
        1. Load calibration labels and segment identifiers.
        2. Iterate over saved calibration images.
        3. Skip images without matching label information.
        4. Detect body landmarks using MediaPipe Pose.
        5. Skip frames in which no valid pose is detected.
        6. Extract posture-related geometric features.
        7. Save annotated landmark images for visual verification.
        8. Store all valid samples in the output dataset.
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

            # Skip frames in which no pose is detected.
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