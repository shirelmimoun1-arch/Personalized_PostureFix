# ============================================================
# LIVE MONITORING – PERSONALIZED POSTURE INFERENCE
# ============================================================
#
# Purpose:
#   Use the trained personal KNN model to classify posture
#   from the webcam in real time.
#
# Flow:
#   1. Open webcam
#   2. Extract MediaPipe landmarks from each frame
#   3. Compute the same features used during training
#   4. Apply imputer
#   5. Apply scaler
#   6. Predict Good / Bad using the personal KNN model
#   7. Apply smoothing logic
#   8. Display live feedback
# ============================================================

import cv2
import joblib
import numpy as np
from collections import deque
import time
import mediapipe as mp
import pandas as pd

from features.feature_extraction import extract_math_features
from features.posture_feedback import PostureFeedback


# ============================================================
# CONFIGURATION
# ============================================================

FPS = 30 # assumed processing rate
WINDOW_SEC = 2.0
WINDOW_SIZE = int(FPS * WINDOW_SEC) # 60 frames, ~2 sec at 30 FPS

BAD_ON_RATIO = 0.7
GOOD_RECOVERY_FRAMES = 20 # ~0.67 sec at 30 FPS


# ============================================================
# STATE
# ============================================================

pred_window = deque(maxlen=WINDOW_SIZE)
feedback_window = deque(maxlen=20) # 0.67 seconds at 30 FPS

stable_state = "Good"
stable_state_since = time.time()
good_streak = 0


# ============================================================
# LABEL + STATE HELPERS
# ============================================================

def normalize_label(label):
    """
    Convert model output into standard Good / Bad labels.
    """

    label = str(label).strip().lower()

    if "bad" in label or label == "0":
        return "Bad"

    if "good" in label or label == "1":
        return "Good"

    return "Unknown"


def state_color(state):
    """
    Return display color according to posture state.
    """

    if state == "Good":
        return (0, 255, 0)

    if state == "Bad":
        return (0, 0, 255)

    return (255, 255, 255)


def update_stable_state(frame_label):
    """
    Smooth frame-by-frame predictions.

    Good → Bad:
        Requires enough Bad frames in the recent window.

    Bad → Good:
        Requires a short streak of Good frames.
    """

    global stable_state, stable_state_since, good_streak

    label = normalize_label(frame_label)

    if label == "Unknown":
        bad_count = sum(1 for x in pred_window if x == "Bad")

        if len(pred_window) == 0:
            bad_ratio = 0.0
        else:
            bad_ratio = bad_count / len(pred_window)

        return stable_state, bad_ratio

    pred_window.append(label)

    bad_count = sum(1 for x in pred_window if x == "Bad")
    bad_ratio = bad_count / len(pred_window)

    if bad_ratio >= BAD_ON_RATIO and stable_state != "Bad":
        stable_state = "Bad"
        stable_state_since = time.time()
        good_streak = 0
        return stable_state, bad_ratio

    if stable_state == "Bad":
        if label == "Good":
            good_streak += 1
        else:
            good_streak = 0

        if good_streak >= GOOD_RECOVERY_FRAMES:
            stable_state = "Good"
            stable_state_since = time.time()
            good_streak = 0

    return stable_state, bad_ratio


# ============================================================
# FEATURE EXTRACTION FROM LIVE FRAME
# ============================================================

def extract_features_from_frame(frame, pose_live, feature_names):
    """
    Extract posture features from one live webcam frame.
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_live.process(rgb)

    if not result.pose_landmarks:
        return {name: np.nan for name in feature_names}

    features = extract_math_features(result.pose_landmarks.landmark)

    return {
        name: features.get(name, np.nan)
        for name in feature_names
    }


# ============================================================
# MODEL PREDICTION
# ============================================================

def too_many_missing(x, threshold=0.4):
    """
    Avoid prediction when too many features are missing.
    """

    return np.mean(np.isnan(x)) > threshold


def predict_from_features(features_dict, imputer, scaler, model, feature_names):
    """
    Apply the same pipeline used during training:
        features → imputer → scaler → KNN prediction

    The KNN model was trained with numeric labels:
        0 = Bad
        1 = Good
    """

    x_df = pd.DataFrame(
        [[features_dict[name] for name in feature_names]],
        columns=feature_names
    )

    x = x_df.values

    if too_many_missing(x):
        return "Unknown", None

    x_imputed = imputer.transform(x_df)
    x_scaled = scaler.transform(x_imputed)

    label_num = model.predict(x_scaled)[0]
    label = "Bad" if int(label_num) == 0 else "Good"

    prob_bad = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_scaled)[0]
        classes = list(model.classes_)

        if 0 in classes:
            prob_bad = float(probs[classes.index(0)])

    return label, prob_bad

def most_common_feedback(feedback_window):
    """
    Return the most common feedback message in the recent window.
    """

    if len(feedback_window) == 0:
        return ""

    counts = {}

    for msg in feedback_window:
        if not msg:
            continue

        counts[msg] = counts.get(msg, 0) + 1

    if not counts:
        return ""

    return max(counts, key=counts.get)

# ============================================================
# MAIN LIVE MONITORING FUNCTION
# ============================================================

def run_live_monitoring(model_path, calibration_dataset_path):
    """
    Run personalized live posture monitoring.
    """

    global stable_state, stable_state_since, good_streak

    pred_window.clear()
    feedback_window.clear()

    stable_state = "Good"
    stable_state_since = time.time()
    good_streak = 0

    pack = joblib.load(model_path)

    imputer = pack["imputer"]
    scaler = pack["scaler"]
    model = pack["model"]
    feature_names = pack["feature_names"]
    classes = pack.get("classes", None)

    print("\nLoaded personal model:")
    print(model_path)
    print("Number of features:", len(feature_names))
    print("Classes:", classes)

    feedback_engine = PostureFeedback(
        calibration_dataset_path=calibration_dataset_path
    )

    cap = cv2.VideoCapture(0) # 1 is for start monitoring from webcam

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose_live:

        WINDOW_NAME = "Personalized Live Posture Monitor"

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 900, 650)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)  # make the video mirror the user

            features = extract_features_from_frame(
                frame=frame,
                pose_live=pose_live,
                feature_names=feature_names
            )

            frame_label, prob_bad = predict_from_features(
                features_dict=features,
                imputer=imputer,
                scaler=scaler,
                model=model,
                feature_names=feature_names
            )

            stable, bad_ratio = update_stable_state(frame_label)

            feedback_message = ""

            if stable == "Bad":
                if frame_label == "Bad":
                    current_feedback = feedback_engine.explain(features)
                    feedback_window.append(current_feedback)

                # keep last common feedback during short Good flickers
                feedback_message = most_common_feedback(feedback_window)

                # clear only when the user is really recovering
                if good_streak >= GOOD_RECOVERY_FRAMES // 2:
                    feedback_window.clear()
                    feedback_message = ""

            else:
                feedback_window.clear()

            elapsed_sec = time.time() - stable_state_since
            color = state_color(stable)

            cv2.putText(
                frame,
                f"Frame: {frame_label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Stable: {stable} | bad_ratio={bad_ratio:.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                3
            )

            cv2.putText(
                frame,
                f"Time in state: {elapsed_sec:.1f}s",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            if prob_bad is not None:
                cv2.putText(
                    frame,
                    f"KNN Bad score: {prob_bad:.2f}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            if feedback_message:
                cv2.putText(
                    frame,
                    feedback_message,
                    (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()