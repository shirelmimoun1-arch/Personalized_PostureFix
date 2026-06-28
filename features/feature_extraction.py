# ============================================================
# FEATURE EXTRACTION – SHARED BY TRAINING AND LIVE INFERENCE
# ============================================================

import numpy as np
import mediapipe as mp


mp_pose = mp.solutions.pose


def wrap_to_pi(angle):
    """
    Wrap angle in radians to range [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def extract_math_features(landmarks):
    """
    Extract posture-related geometric features and raw landmark data.

    Missing landmarks are represented as NaN.
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
        "shoulderTilt": np.nan,
        # "abs_shoulderTilt": np.nan,

        "torsoRotation": np.nan,
        # "abs_torsoRotation": np.nan,
        "shoulderWidth": np.nan,

        "theta_shoulders": np.nan,
        "thetaNeck": np.nan,
        "thetaNeck_rel": np.nan,
        # "abs_thetaNeck_rel": np.nan,

        "theta_RS_proj": np.nan,
        "theta_LS_proj": np.nan,
        "theta_RS_proj_relShoulders": np.nan,
        "theta_LS_proj_relShoulders": np.nan,

        "theta_ears": np.nan,
        "theta_ears_rel": np.nan,
        # "abs_theta_ears_rel": np.nan,

        "theta_LEar_LS": np.nan,
        "theta_LEar_LS_relShoulders": np.nan,
        # "abs_theta_LEar_LS_relShoulders": np.nan,

        "theta_REar_RS": np.nan,
        "theta_REar_RS_relShoulders": np.nan,
        # "abs_theta_REar_RS_relShoulders": np.nan,
    }

    add_landmark_fields(res, "nose", NOSE)
    add_landmark_fields(res, "leftEar", LEAR)
    add_landmark_fields(res, "rightEar", REAR)
    add_landmark_fields(res, "leftShoulder", LS)
    add_landmark_fields(res, "rightShoulder", RS)
    add_landmark_fields(res, "leftEye", LEYE)
    add_landmark_fields(res, "rightEye", REYE)

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

            res["shoulderWidth"] = shoulder_width
            res["shoulderTilt"] = (LS.y - RS.y) / shoulder_width
            res["torsoRotation"] = RS.z - LS.z

            # res["abs_shoulderTilt"] = abs(res["shoulderTilt"])
            # res["abs_torsoRotation"] = abs(res["torsoRotation"])

            theta_shoulders = np.arctan2(RS.y - LS.y, RS.x - LS.x)
            res["theta_shoulders"] = np.degrees(theta_shoulders)

            if theta_ears is not None:
                res["theta_ears_rel"] = np.degrees(
                    wrap_to_pi(theta_ears - theta_shoulders)
                )
                # res["abs_theta_ears_rel"] = abs(res["theta_ears_rel"])

            if theta_L is not None:
                res["theta_LEar_LS_relShoulders"] = np.degrees(
                    wrap_to_pi(theta_L - theta_shoulders)
                )
                # res["abs_theta_LEar_LS_relShoulders"] = abs(
                #     res["theta_LEar_LS_relShoulders"]
                # )

            if theta_R is not None:
                res["theta_REar_RS_relShoulders"] = np.degrees(
                    wrap_to_pi(theta_R - theta_shoulders)
                )
                # res["abs_theta_REar_RS_relShoulders"] = abs(
                #     res["theta_REar_RS_relShoulders"]
                # )

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
                # res["abs_thetaNeck_rel"] = abs(res["thetaNeck_rel"])

                res["headForwardDepth"] = (Sz - NOSE.z) / shoulder_width
                res["headHeight"] = (Sy - NOSE.y) / shoulder_width

    return res