# ============================================================
# POSTURE FEATURE EXTRACTION
# SHARED BY TRAINING AND LIVE INFERENCE
# ============================================================
#
# Purpose:
#   Extract a consistent set of geometric posture descriptors from
#   MediaPipe Pose landmarks for both offline model training and
#   real-time inference.
#
# Input:
#   - MediaPipe Pose landmarks corresponding to a single video frame.
#
# Output:
#   A dictionary containing:
#       1. Engineered geometric posture features.
#       2. Raw normalized landmark coordinates and visibility values
#          for the selected upper-body landmarks
#
# Feature categories:
#   - Head position relative to the shoulder center.
#   - Head height and forward displacement.
#   - Horizontal head displacement and head rotation.
#   - Shoulder width, tilt, and depth asymmetry.
#   - Eye and ear orientation relative to the shoulders.
#   - Ear-to-shoulder geometric relationships.
#   - Raw landmark coordinates for the nose, eyes, ears, and shoulders.
#
# Normalization:
#   Several geometric measurements are normalized by shoulder width
#   in order to reduce sensitivity to user-camera distance and body size.
#
# Missing data:
#   Landmarks with visibility <= 0.1 are treated as unavailable.
#   Features that cannot be computed because one or more required
#   landmarks are unavailable are represented by NaN.
#
# Consistency requirement:
#   This module is used by both dataset generation and live inference.
#   Maintaining a single shared feature-extraction implementation
#   prevents discrepancies between the training and deployment pipelines.
# ============================================================

import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose


def wrap_to_pi(angle):
    """
    Normalize an angle to the interval [-pi, pi).

    Args:
        angle (float):
            Angle in radians.

    Returns:
        float:
            Equivalent angle wrapped to the interval [-pi, pi).

    Notes:
        Angle wrapping is used when computing relative orientations
        between anatomical landmark pairs. It prevents discontinuities
        caused by equivalent angular representations that differ by
        integer multiples of 2*pi.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def extract_math_features(landmarks):
    """
    Extract posture-related geometric features from MediaPipe Pose landmarks.

    The function converts selected upper-body landmarks into a fixed set
    of interpretable numerical descriptors used by both the personalized
    model-training pipeline and the real-time inference pipeline.

    The extracted representation includes geometric relationships between
    the nose, eyes, ears, and shoulders, together with the raw normalized
    MediaPipe coordinates of these landmarks.

    Args:
        landmarks:
            Sequence of MediaPipe Pose landmarks for a single frame.
            Each landmark contains normalized x, y, and z coordinates
            together with a visibility score.

    Returns:
        dict:
            Dictionary containing engineered posture features and raw
            landmark fields.

            Engineered features include:

            Head position and orientation:
                - headForwardDepth
                - headHeight
                - headTurn
                - headHorizontalOffsetNorm
                - eyeCenterHeightNorm
                - earCenterHeightNorm
                - earDepthDifferenceNorm

            Shoulder geometry:
                - shoulderWidth
                - shoulderTilt
                - torsoRotation
                - theta_shoulders

            Eye and ear orientation:
                - eyeWidth
                - eyeTiltRelShoulders
                - theta_ears
                - theta_ears_rel

            Neck and shoulder-relative angles:
                - thetaNeck
                - thetaNeck_rel
                - theta_RS_proj
                - theta_LS_proj
                - theta_RS_proj_relShoulders
                - theta_LS_proj_relShoulders

            Ear-to-shoulder geometry:
                - theta_LEar_LS
                - theta_LEar_LS_relShoulders
                - theta_REar_RS
                - theta_REar_RS_relShoulders
                - leftEarShoulderDistance
                - rightEarShoulderDistance

            Raw landmark fields are stored for:
                - nose
                - leftEar
                - rightEar
                - leftShoulder
                - rightShoulder
                - leftEye
                - rightEye

            For each raw landmark, the following values are stored:
                - x
                - y
                - z
                - visibility

    Notes:
        MediaPipe coordinates are normalized image coordinates. The x and y
        coordinates describe horizontal and vertical landmark location,
        respectively, while z represents relative depth in the MediaPipe
        pose coordinate system.

        Landmarks whose visibility score is less than or equal to 0.1 are
        treated as unavailable.

        When a required landmark is unavailable, the corresponding feature
        remains NaN rather than being estimated from incomplete geometry.

        Several features are normalized by shoulder width in order to reduce
        sensitivity to body scale and distance from the camera.

        Relative angular features are wrapped to [-pi, pi) before conversion
        to degrees. This avoids angular discontinuities when subtracting
        two orientations.

        Raw landmark coordinates are retained primarily for analysis,
        debugging, and reproducibility. The downstream classifier may use
        only a selected subset of the returned fields.
    """
    def get_pt(idx):
        pt = landmarks[idx.value]
        return pt if pt.visibility > 0.1 else None

    def add_landmark_fields(res, prefix, lm):
        res[f"{prefix}_x"] = np.nan if lm is None else float(lm.x)
        res[f"{prefix}_y"] = np.nan if lm is None else float(lm.y)
        res[f"{prefix}_z"] = np.nan if lm is None else float(lm.z)
        res[f"{prefix}_vis"] = np.nan if lm is None else float(lm.visibility)

    LS = get_pt(mp_pose.PoseLandmark.LEFT_SHOULDER)
    RS = get_pt(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    NOSE = get_pt(mp_pose.PoseLandmark.NOSE)
    LEAR = get_pt(mp_pose.PoseLandmark.LEFT_EAR)
    REAR = get_pt(mp_pose.PoseLandmark.RIGHT_EAR)
    LEYE = get_pt(mp_pose.PoseLandmark.LEFT_EYE)
    REYE = get_pt(mp_pose.PoseLandmark.RIGHT_EYE)

    res = {
        "headForwardDepth": np.nan,
        "headHeight": np.nan,
        "headTurn": np.nan,
        "eyeWidth": np.nan,

        "headHorizontalOffsetNorm": np.nan,
        "eyeCenterHeightNorm": np.nan,
        "earCenterHeightNorm": np.nan,
        "eyeTiltRelShoulders": np.nan,
        "earDepthDifferenceNorm": np.nan,

        "shoulderTilt": np.nan,

        "torsoRotation": np.nan,
        "shoulderWidth": np.nan,

        "theta_shoulders": np.nan,
        "thetaNeck": np.nan,
        "thetaNeck_rel": np.nan,

        "theta_RS_proj": np.nan,
        "theta_LS_proj": np.nan,
        "theta_RS_proj_relShoulders": np.nan,
        "theta_LS_proj_relShoulders": np.nan,

        "theta_ears": np.nan,
        "theta_ears_rel": np.nan,

        "theta_LEar_LS": np.nan,
        "theta_LEar_LS_relShoulders": np.nan,

        "theta_REar_RS": np.nan,
        "theta_REar_RS_relShoulders": np.nan,

        "leftEarShoulderDistance": np.nan,
        "rightEarShoulderDistance": np.nan,
    }

    add_landmark_fields(res, "nose", NOSE)
    add_landmark_fields(res, "leftEar", LEAR)
    add_landmark_fields(res, "rightEar", REAR)
    add_landmark_fields(res, "leftShoulder", LS)
    add_landmark_fields(res, "rightShoulder", RS)
    add_landmark_fields(res, "leftEye", LEYE)
    add_landmark_fields(res, "rightEye", REYE)

    if NOSE and LEYE and REYE:
        eye_width = abs(REYE.x - LEYE.x)

        if eye_width > 1e-6:
            eye_center_x = (LEYE.x + REYE.x) / 2
            res["eyeWidth"] = eye_width
            res["headTurn"] = (NOSE.x - eye_center_x) / eye_width

    theta_ears = None
    theta_L = None
    theta_R = None

    if LEAR and REAR:
        theta_ears = np.arctan2(REAR.y - LEAR.y, REAR.x - LEAR.x)
        res["theta_ears"] = np.degrees(theta_ears)

    if LS and LEAR:
        theta_L = np.arctan2(LEAR.x - LS.x, LEAR.y - LS.y)
        res["theta_LEar_LS"] = np.degrees(theta_L)

    if RS and REAR:
        theta_R = np.arctan2(REAR.x - RS.x, REAR.y - RS.y)
        res["theta_REar_RS"] = np.degrees(theta_R)

    if LS and RS:
        shoulder_width = np.hypot(RS.x - LS.x, RS.y - LS.y)

        if shoulder_width > 0:
            Sx = (LS.x + RS.x) / 2
            Sy = (LS.y + RS.y) / 2
            Sz = (LS.z + RS.z) / 2

            if NOSE:
                res["headHorizontalOffsetNorm"] = (
                        (NOSE.x - Sx)
                        / shoulder_width
                )

            if LEYE and REYE:
                eye_center_y = (
                                       LEYE.y + REYE.y
                               ) / 2

                res["eyeCenterHeightNorm"] = (
                        (Sy - eye_center_y)
                        / shoulder_width
                )

            if LEAR and REAR:
                ear_center_y = (
                                       LEAR.y + REAR.y
                               ) / 2

                res["earCenterHeightNorm"] = (
                        (Sy - ear_center_y)
                        / shoulder_width
                )

                res["earDepthDifferenceNorm"] = (
                        (REAR.z - LEAR.z)
                        / shoulder_width
                )

            res["shoulderWidth"] = shoulder_width
            res["shoulderTilt"] = (LS.y - RS.y) / shoulder_width
            res["torsoRotation"] = RS.z - LS.z

            if LEAR:
                res["leftEarShoulderDistance"] = (
                        np.hypot(
                            LEAR.x - LS.x,
                            LEAR.y - LS.y
                        ) / shoulder_width
                )

            if REAR:
                res["rightEarShoulderDistance"] = (
                        np.hypot(
                            REAR.x - RS.x,
                            REAR.y - RS.y
                        ) / shoulder_width
                )

            theta_shoulders = np.arctan2(RS.y - LS.y, RS.x - LS.x)
            res["theta_shoulders"] = np.degrees(theta_shoulders)

            if LEYE and REYE:
                theta_eyes = np.arctan2(
                    REYE.y - LEYE.y,
                    REYE.x - LEYE.x
                )

                res["eyeTiltRelShoulders"] = np.degrees(
                    wrap_to_pi(
                        theta_eyes
                        - theta_shoulders
                    )
                )

            if theta_ears is not None:
                res["theta_ears_rel"] = np.degrees(
                    wrap_to_pi(theta_ears - theta_shoulders)
                )

            if theta_L is not None:
                res["theta_LEar_LS_relShoulders"] = np.degrees(
                    wrap_to_pi(theta_L - theta_shoulders)
                )

            if theta_R is not None:
                res["theta_REar_RS_relShoulders"] = np.degrees(
                    wrap_to_pi(theta_R - theta_shoulders)
                )

            theta_RS_proj = np.arctan2(RS.x - Sx, RS.y - Sy)
            theta_LS_proj = np.arctan2(LS.x - Sx, LS.y - Sy)

            res["theta_RS_proj"] = np.degrees(theta_RS_proj)
            res["theta_LS_proj"] = np.degrees(theta_LS_proj)

            res["theta_RS_proj_relShoulders"] = np.degrees(
                wrap_to_pi(theta_RS_proj - theta_shoulders)
            )
            res["theta_LS_proj_relShoulders"] = np.degrees(
                wrap_to_pi(theta_LS_proj - theta_shoulders)
            )

            if NOSE:
                theta_neck = np.arctan2(NOSE.x - Sx, NOSE.y - Sy)
                res["thetaNeck"] = np.degrees(theta_neck)
                res["thetaNeck_rel"] = np.degrees(
                    wrap_to_pi(theta_neck - theta_shoulders)
                )

                res["headForwardDepth"] = (Sz - NOSE.z) / shoulder_width
                res["headHeight"] = (Sy - NOSE.y) / shoulder_width

    return res