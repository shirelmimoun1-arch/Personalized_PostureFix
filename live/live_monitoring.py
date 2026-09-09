# ============================================================
# LIVE MONITORING – PERSONALIZED POSTURE INFERENCE
# ============================================================
#
# Purpose:
#   Use the trained personal KNN model to classify posture
#   from the webcam in real time.
# Flow:
#   1. Open the webcam and acquire mirrored video frames.
#   2. Extract MediaPipe Pose landmarks.
#   3. Compute the same geometric features used during training.
#   4. Apply the fitted imputer and scaler.
#   5. Predict frame-level Good / Bad posture using the personalized KNN.
#   6. Stabilize frame predictions using time-based posture-state logic.
#   7. For stable Bad posture, generate model-aware corrective feedback.
#   8. Evaluate supplementary personalized diagnostics such as shoulder
#      asymmetry without modifying the KNN prediction.
#   9. Stabilize corrective messages using a separate temporal
#      feedback window and hysteresis mechanism.
#  10. Display the stabilized state and corrective feedback.
# ============================================================

import cv2
import joblib
import numpy as np
from collections import deque
import time
import mediapipe as mp
import pandas as pd

from features.feature_extraction import extract_math_features
from features.posture_feedback_model_aware import PostureFeedback


# ============================================================
# CONFIGURATION
# ============================================================

# Duration of the recent prediction history used to determine
# whether poor posture is persistent enough to trigger a Bad state.
BAD_WINDOW_SEC = 2.0

# Minimum fraction of Bad predictions within the recent time window
# required to transition from Good to Bad.
BAD_ON_RATIO = 0.7

# Continuous duration of Good predictions required to recover
# from a stable Bad state back to Good.
GOOD_RECOVERY_SEC = 0.67

# Default camera device. The index may need to be changed on systems
# containing multiple physical or virtual cameras.
CAMERA_INDEX = 0

# Use a widescreen 16:9 capture format while keeping the
# resolution moderate enough for real-time pose estimation.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Minimum amount of prediction history required before a Bad
# state can be activated.
MIN_BAD_OBSERVATION_SEC = 1.0

# Duration of the recent feedback history used to stabilize the
# corrective message displayed to the user. A longer window prevents
# local KNN explanation changes from causing rapid message switching.
FEEDBACK_WINDOW_SEC = 2.0

# Higher MediaPipe Pose model complexity is used to favor landmark
# accuracy over computational speed. On lower-performance hardware,
# this setting may reduce the effective processing frame rate.
POSE_MODEL_COMPLEXITY = 2

# Duration for which a new feedback candidate must remain dominant
# before replacing the currently displayed corrective message.
FEEDBACK_SWITCH_SEC = 1.5

# ============================================================
# STATE
# ============================================================

# Each entry stores:
#     (timestamp, normalized_prediction)
#
# Unlike a frame-count-based window, this structure stores all
# predictions produced during the most recent BAD_WINDOW_SEC seconds.
pred_window = deque()

# Each entry stores:
#     (timestamp, corrective_message)
#
# The window contains recently generated feedback messages and is
# used to determine the dominant explanation before temporal
# feedback hysteresis is applied.
feedback_window = deque()

displayed_feedback = ""
feedback_candidate = ""
feedback_candidate_since = None

stable_state = "Good"
stable_state_since = time.monotonic()

# Timestamp marking the beginning of the current uninterrupted
# sequence of Good predictions while recovering from a Bad state.
good_streak_start = None

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

def draw_multiline_text(
        frame,
        text,
        start_x,
        start_y,
        max_width,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=0.7,
        color=(0, 0, 0),
        thickness=2,
        line_spacing=10
):
    """
    Draw text on multiple lines while keeping each line within max_width.

    Words are added sequentially until the next word would exceed the
    permitted width. The remaining text is then drawn on the next line.
    """

    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = (
            word if not current_line
            else current_line + " " + word
        )

        (text_width, text_height), _ = cv2.getTextSize(
            test_line,
            font,
            font_scale,
            thickness
        )

        if text_width <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)

            current_line = word

    if current_line:
        lines.append(current_line)

    y = start_y

    for line in lines:
        cv2.putText(
            frame,
            line,
            (start_x, y),
            font,
            font_scale,
            color,
            thickness
        )

        (_, text_height), _ = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness
        )

        y += text_height + line_spacing


def update_stable_state(frame_label):
    """
    Update the temporally stabilized posture state using elapsed time.

    Raw frame-level predictions may fluctuate because of landmark noise
    or short user movements. Temporal smoothing is therefore applied
    using timestamps rather than a fixed number of frames, making the
    behavior less dependent on the processing frame rate.

    Good -> Bad:
        Predictions from the most recent BAD_WINDOW_SEC seconds are
        considered. The state changes to Bad when at least BAD_ON_RATIO
        of these predictions are classified as Bad.

    Bad -> Good:
        Recovery requires continuously receiving Good predictions for
        at least GOOD_RECOVERY_SEC seconds.

    Frames classified as Unknown are not added to the prediction history.
    However, an Unknown frame interrupts an ongoing Good recovery interval,
    since recovery requires continuously observed Good predictions.

    Returns
    -------
    stable_state : str
        Current temporally stabilized posture state.

    bad_ratio : float
        Fraction of Bad predictions in the current temporal window.
    """

    global stable_state, stable_state_since, good_streak_start

    label = normalize_label(frame_label)
    now = time.monotonic()

    # --------------------------------------------------------
    # Remove predictions older than the configured time window
    # --------------------------------------------------------

    while pred_window and now - pred_window[0][0] > BAD_WINDOW_SEC:
        pred_window.popleft()

    # --------------------------------------------------------
    # Unknown frames are ignored
    # --------------------------------------------------------

    if label == "Unknown":

        # Unknown frames do not contribute to the Bad ratio.
        # During recovery, however, loss of a reliable prediction
        # interrupts the continuous Good interval.
        if stable_state == "Bad":
            good_streak_start = None

        if len(pred_window) == 0:
            bad_ratio = 0.0
        else:
            bad_count = sum(
                1 for _, prediction in pred_window
                if prediction == "Bad"
            )
            bad_ratio = bad_count / len(pred_window)

        return stable_state, bad_ratio

    # --------------------------------------------------------
    # Store current prediction together with its timestamp
    # --------------------------------------------------------

    pred_window.append((now, label))

    bad_count = sum(
        1 for _, prediction in pred_window
        if prediction == "Bad"
    )

    bad_ratio = bad_count / len(pred_window)

    # --------------------------------------------------------
    # Good -> Bad transition
    # --------------------------------------------------------

    if len(pred_window) >= 2:
        observed_duration = (
                pred_window[-1][0] - pred_window[0][0]
        )
    else:
        observed_duration = 0.0

    if (
            stable_state != "Bad"
            and observed_duration >= MIN_BAD_OBSERVATION_SEC
            and bad_ratio >= BAD_ON_RATIO
    ):
        stable_state = "Bad"
        stable_state_since = now
        good_streak_start = None

        return stable_state, bad_ratio

    # --------------------------------------------------------
    # Bad -> Good recovery
    # --------------------------------------------------------

    if stable_state == "Bad":

        if label == "Good":

            if good_streak_start is None:
                good_streak_start = now

            good_duration = now - good_streak_start

            if good_duration >= GOOD_RECOVERY_SEC:
                stable_state = "Good"
                stable_state_since = now
                good_streak_start = None

        else:
            # Any Bad prediction interrupts the continuous Good period.
            good_streak_start = None

    return stable_state, bad_ratio

# ============================================================
# FEATURE EXTRACTION FROM LIVE FRAME
# ============================================================

def extract_features_from_frame(frame, pose_live, feature_names):
    """
    Extract model and supplementary feedback features from one webcam frame.

    MediaPipe Pose is applied to the current frame and the same geometric
    feature-extraction function used during dataset construction is reused.
    Only the ordered features required by the trained KNN are retained for
    classification.

    ``shoulderTilt`` is additionally retained as a supplementary
    interpretable feature for personalized shoulder-asymmetry feedback.
    It is not included in the KNN input unless explicitly present in
    ``feature_names``.

    If no pose is detected, all required features are returned as NaN so that
    the prediction stage can reject unreliable frames as Unknown.
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose_live.process(rgb)

    if not result.pose_landmarks:
        missing_features = {
            name: np.nan
            for name in feature_names
        }

        missing_features["shoulderTilt"] = np.nan

        return missing_features

    features = extract_math_features(
        result.pose_landmarks.landmark
    )

    selected_features = {
        name: features.get(name, np.nan)
        for name in feature_names
    }

    # Supplementary interpretable feature used only for feedback.
    # It is not passed to the KNN classifier.
    selected_features["shoulderTilt"] = features.get(
        "shoulderTilt",
        np.nan
    )

    return selected_features


# ============================================================
# MODEL PREDICTION
# ============================================================

def too_many_missing(x, threshold=0.4):
    """
    Determine whether a feature vector contains insufficient pose data.

    A frame is rejected when more than ``threshold`` of its features are
    missing. This prevents the classifier from relying predominantly on
    imputed values when pose landmarks are poorly detected.

    Parameters
    ----------
    x : array-like
        Feature vector to evaluate.
    threshold : float, default=0.4
        Maximum tolerated fraction of missing feature values.

    Returns
    -------
    bool
        True when the missing-value ratio exceeds the threshold.
    """
    return np.mean(np.isnan(x)) > threshold


def predict_from_features(features_dict, imputer, scaler, model, feature_names):
    """
    Generate a posture prediction from an extracted feature vector.

    The feature vector is arranged according to the feature ordering stored
    with the trained personalized model. Missing values are handled using
    the imputer fitted during training, after which the vector is
    standardized using the previously fitted scaler and passed to the KNN
    classifier.

    Frames containing more than the permitted fraction of missing features
    are rejected and assigned the ``Unknown`` label rather than being
    classified from unreliable input.

    Parameters
    ----------
    features_dict : dict
        Mapping from feature names to values extracted from the current frame.
    imputer :
        Imputation transformer fitted on the training data.
    scaler :
        Feature scaler fitted on the training data.
    model :
        Trained personalized KNN classifier.
    feature_names : sequence of str
        Ordered feature names expected by the trained model.

    Returns
    -------
    label : str
        Predicted posture label: ``"Good"``, ``"Bad"``, or ``"Unknown"``.
    prob_bad : float or None
        KNN class-support score for the Bad class when ``predict_proba``
        is available.
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
    Return the dominant corrective message in the recent feedback window.

    Empty feedback entries are ignored. The returned message is subsequently
    passed to the temporal hysteresis mechanism before being displayed.
    """
    counts = {}

    for _, msg in feedback_window:
        if not msg:
            continue

        counts[msg] = counts.get(msg, 0) + 1

    if not counts:
        return ""

    return max(counts, key=counts.get)

def update_stable_feedback(candidate_feedback):
    """
    Stabilize the corrective feedback using temporal hysteresis.

    The currently displayed message is preserved while short-lived
    alternatives are ignored. A new message replaces it only after
    remaining the dominant feedback candidate continuously for
    FEEDBACK_SWITCH_SEC seconds.

    Parameters
    ----------
    candidate_feedback : str
        Most common feedback message in the recent feedback window.

    Returns
    -------
    str
        Temporally stabilized feedback message.
    """

    global displayed_feedback
    global feedback_candidate
    global feedback_candidate_since

    now = time.monotonic()

    # No valid new candidate -> keep the current displayed message.
    if not candidate_feedback:
        return displayed_feedback

    # No feedback is currently displayed.
    # The first reliable candidate can be shown immediately.
    if not displayed_feedback:
        displayed_feedback = candidate_feedback
        feedback_candidate = ""
        feedback_candidate_since = None
        return displayed_feedback

    # Current candidate agrees with what is already displayed.
    if candidate_feedback == displayed_feedback:
        feedback_candidate = ""
        feedback_candidate_since = None
        return displayed_feedback

    # A different candidate has appeared.
    if candidate_feedback != feedback_candidate:
        feedback_candidate = candidate_feedback
        feedback_candidate_since = now
        return displayed_feedback

    # The same alternative candidate is still dominant.
    if feedback_candidate_since is not None:
        candidate_duration = now - feedback_candidate_since

        if candidate_duration >= FEEDBACK_SWITCH_SEC:
            displayed_feedback = feedback_candidate
            feedback_candidate = ""
            feedback_candidate_since = None

    return displayed_feedback

# ============================================================
# MAIN LIVE MONITORING FUNCTION
# ============================================================

def run_live_monitoring(model_path, calibration_dataset_path):
    """
    Run personalized real-time posture monitoring.

    The function loads the trained personalized model and preprocessing
    objects, opens the webcam, performs real-time posture inference,
    applies temporal state stabilization, generates corrective feedback,
    and displays the resulting posture state to the user.

    Parameters
    ----------
    model_path : str or Path
        Path to the serialized personalized model package produced by
        the training pipeline.

    calibration_dataset_path : str or Path
        Path to the user's calibration feature dataset. The dataset is
        used only to construct supplementary personalized feedback
        references, such as the Good-posture shoulder-alignment reference,
        and does not modify the trained KNN classifier.

    Notes
    -----
    Camera and OpenCV window resources are released in a ``finally`` block
    so cleanup is performed even if an exception occurs during monitoring.
    """
    global stable_state, stable_state_since, good_streak_start
    global displayed_feedback, feedback_candidate, feedback_candidate_since

    pred_window.clear()
    feedback_window.clear()

    displayed_feedback = ""
    feedback_candidate = ""
    feedback_candidate_since = None

    stable_state = "Good"
    stable_state_since = time.monotonic()
    good_streak_start = None

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
        pack,
        calibration_dataset_path=calibration_dataset_path
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)

    # A moderate capture resolution limits computational cost while preserving
    # sufficient spatial information for upper-body pose estimation.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open webcam.")

    try:
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=POSE_MODEL_COMPLEXITY,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose_live:

            WINDOW_NAME = "Personalized Live Posture Monitor"

            cv2.namedWindow(WINDOW_NAME)

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # Mirror the display for the user.

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
                    now = time.monotonic()

                    # Remove feedback samples that are older than the configured
                    # temporal feedback window, even when the current frame does
                    # not produce a new Bad explanation.
                    while (
                            feedback_window
                            and now - feedback_window[0][
                                0] > FEEDBACK_WINDOW_SEC
                    ):
                        feedback_window.popleft()

                    if frame_label == "Bad":
                        current_feedback = feedback_engine.explain(features)

                        feedback_window.append(
                            (now, current_feedback)
                        )

                    candidate_feedback = most_common_feedback(
                        feedback_window
                    )

                    feedback_message = update_stable_feedback(
                        candidate_feedback
                    )

                    # Clear feedback after recovery has been sustained for
                    # half of the required Good recovery duration.
                    if good_streak_start is not None:
                        recovery_duration = (
                            time.monotonic() - good_streak_start
                        )

                        if recovery_duration >= GOOD_RECOVERY_SEC / 2:
                            feedback_window.clear()

                            displayed_feedback = ""
                            feedback_candidate = ""
                            feedback_candidate_since = None

                            feedback_message = ""


                else:

                    feedback_window.clear()

                    displayed_feedback = ""

                    feedback_candidate = ""

                    feedback_candidate_since = None

                elapsed_sec = time.monotonic() - stable_state_since
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
                    draw_multiline_text(
                        frame=frame,
                        text=feedback_message,
                        start_x=20,
                        start_y=180,
                        max_width=frame.shape[1] - 40,
                        font_scale=0.7,
                        color=(0, 0, 0),
                        thickness=2,
                        line_spacing=10
                    )

                cv2.imshow(WINDOW_NAME, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
