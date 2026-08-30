import numpy as np
import pandas as pd
from pathlib import Path


class PostureFeedback:
    def __init__(self, calibration_dataset_path: str):
        path = Path(calibration_dataset_path)

        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        label_col = self._find_column(df, ["Label", "label"])
        if label_col is None:
            raise ValueError("Dataset must contain a Label column")

        good_df = df[df[label_col].astype(str).str.lower().str.contains("good")].copy()
        bad_df = df[df[label_col].astype(str).str.lower().str.contains("bad")].copy()

        if good_df.empty:
            raise ValueError("No GOOD samples found in calibration dataset")
        if bad_df.empty:
            raise ValueError("No BAD samples found in calibration dataset")

        excluded_cols = {
            "Image_ID", "image_id",
            "Label", "label",
            "Segment_ID", "segment_id",
            "Frame", "frame",
            "Path", "path"
        }

        self.feature_cols = [
            col for col in good_df.columns
            if col not in excluded_cols
            and pd.api.types.is_numeric_dtype(good_df[col])
        ]

        self.good_mean = good_df[self.feature_cols].mean()
        self.good_std = good_df[self.feature_cols].std().replace(0, 1e-6)
        self.bad_mean = bad_df[self.feature_cols].mean()

    def explain(self, current_features: dict, top_k: int = 1) -> str:
        issues = []

        self._add_asymmetric_shoulders_issue(issues, current_features)

        if not issues:
            self._add_rounded_shoulders_back_issue(issues, current_features)

        if not issues:
            self._add_head_up_issue(issues, current_features)

        if not issues:
            self._add_head_down_issue(issues, current_features)

        if not issues:
            self._add_shoulders_up_issue(issues, current_features)

        lean_issue = self._get_head_lean_issue(current_features)
        turn_issue = self._get_head_turn_issue(current_features)

        if lean_issue is not None and turn_issue is not None:
            lean_score, _ = lean_issue
            turn_score, _ = turn_issue

            if lean_score >= turn_score:
                issues.append(lean_issue)
            else:
                issues.append(turn_issue)

        elif lean_issue is not None:
            issues.append(lean_issue)

        elif turn_issue is not None:
            issues.append(turn_issue)

        if not issues:
            self._add_head_forward_issue(issues, current_features)

        if not issues:
            return "Bad posture detected. Try sitting upright and relaxing your shoulders."

        issues.sort(reverse=True, key=lambda x: x[0])

        messages = []
        for _, message in issues:
            if message not in messages:
                messages.append(message)
            if len(messages) >= top_k:
                break

        return " ".join(messages)

    def _add_rounded_shoulders_back_issue(self, issues, f):
        shoulders_up_score = self._mean_valid([
            self._z_toward_bad(f, "leftShoulder_y", expected_direction=-1),
            self._z_toward_bad(f, "rightShoulder_y", expected_direction=-1),
        ])

        head_down_score = self._mean_valid([
            self._z_toward_bad(f, "nose_y", expected_direction=+1),
            self._z_toward_bad(f, "headHeight", expected_direction=-1),
        ])

        # Need shoulder evidence. Head-down alone is not enough.
        if shoulders_up_score < 1.2:
            return

        # Head can be only mildly lowered.
        if head_down_score < 0.7:
            return

        score = 0.7 * shoulders_up_score + 0.3 * head_down_score

        issues.append(
            (score,
             "Your upper back seems rounded. Try opening your chest and relaxing your shoulders.")
        )

    def _add_head_down_issue(self, issues, f):
        score = self._mean_valid([
            self._z_toward_bad(f, "nose_y", expected_direction=+1),
            self._z_toward_bad(f, "headHeight", expected_direction=-1),
        ])

        if score < 1.5:
            return

        issues.append((score, "Your head or upper body seems lowered. "
                              "Try sitting taller and keeping your gaze level."))

    def _add_head_up_issue(self, issues, f):
        score = self._z_abs_from_good(f, "headHeight")
        delta = self._delta_from_good(f, "headHeight")

        if score is None or delta is None:
            return

        # head higher than normal
        if delta <= 0:
            return

        if score < 1.5:
            return

        issues.append(
            (score, "Your head seems tilted upward.")
        )

    def _add_shoulders_up_issue(self, issues, f):
        left_score = self._z_toward_bad(
            f,
            "leftEarShoulderDistance",
            expected_direction=-1
        )

        right_score = self._z_toward_bad(
            f,
            "rightEarShoulderDistance",
            expected_direction=-1
        )

        valid_scores = [
            score for score in [left_score, right_score]
            if score is not None and not pd.isna(score)
        ]

        # Require both sides to be visible and abnormal.
        # This reduces false feedback caused by an inaccurate ear landmark.
        if len(valid_scores) < 2:
            return

        score = float(np.mean(valid_scores))

        if score < 1.5:
            return

        issues.append((
            score,
            "Your shoulders seem raised toward your ears. Try lowering and relaxing them."
        ))

    def _add_asymmetric_shoulders_issue(self, issues, f):
        score = self._z_abs_from_good(f, "shoulderTilt")
        delta = self._delta_from_good(f, "shoulderTilt")

        if score is None or delta is None:
            return

        if score < 1.2 and abs(delta) < 0.04:
            return

        feedback_score = max(score, 1.5)

        # shoulderTilt = (LS.y - RS.y) / shoulder_width
        # y grows downward
        # delta > 0 means left shoulder lower / right shoulder higher
        if delta > 0:
            issues.append((feedback_score,
                           "Your left shoulder seems higher than your left shoulder."))
        else:
            issues.append((feedback_score,
                           "Your right shoulder seems higher than your right shoulder."))

    def _get_head_lean_issue(self, f):
        score = self._z_abs_from_good(f, "theta_ears_rel")
        delta = self._delta_from_good(f, "theta_ears_rel")

        if score is None or delta is None:
            return None

        if score < 1.2 and abs(delta) < 5.0:
            return None

        feedback_score = max(score, 1.5)

        if delta > 0:
            return (feedback_score,
                    "Your head is leaning too much to the right.")
        else:
            return (feedback_score,
                    "Your head is leaning too much to the left.")

    def _get_head_turn_issue(self, f):
        score = self._z_abs_from_good(f, "headTurn")
        delta = self._delta_from_good(f, "headTurn")

        if score is None or delta is None:
            return None

        if score < 1.2 and abs(delta) < 0.25:
            return None

        feedback_score = max(score, 1.5)

        if delta > 0:
            return (feedback_score, "Your head seems turned to the right.")
        else:
            return (feedback_score, "Your head seems turned to the left.")

    def _add_head_forward_issue(self, issues, f):
        score = self._z_toward_bad(
            f,
            "headForwardDepth",
            expected_direction=+1
        )

        if score is None or score < 1.5:
            return

        issues.append((score, "Your head is too far forward."))

    def _z_toward_bad(self, current_features, feature, expected_direction=None):
        if feature not in current_features:
            return None

        if feature not in self.good_mean.index:
            return None

        value = current_features[feature]

        if value is None or pd.isna(value):
            return None

        good_value = self.good_mean[feature]
        bad_value = self.bad_mean[feature]
        std = self.good_std[feature]

        if pd.isna(good_value) or pd.isna(bad_value) or pd.isna(std):
            return None

        if std <= 0:
            return None

        current_delta = value - good_value
        bad_delta = bad_value - good_value

        if pd.isna(current_delta) or pd.isna(bad_delta):
            return None

        if expected_direction is not None:
            if np.sign(current_delta) != expected_direction:
                return None
        else:
            if abs(bad_delta) < 0.25 * std:
                return None

            if np.sign(current_delta) != np.sign(bad_delta):
                return None

        return self._cap_score(abs(current_delta) / std)

    def _delta_from_good(self, current_features, feature):
        if feature not in current_features:
            return None

        if feature not in self.good_mean.index:
            return None

        value = current_features[feature]

        if value is None or pd.isna(value):
            return None

        good_value = self.good_mean[feature]

        if pd.isna(good_value):
            return None

        return float(value - good_value)

    def _z_abs_from_good(self, current_features, feature):
        if feature not in current_features:
            return None

        if feature not in self.good_mean.index:
            return None

        value = current_features[feature]

        if value is None or pd.isna(value):
            return None

        good_value = self.good_mean[feature]
        std = self.good_std[feature]

        if pd.isna(good_value) or pd.isna(std) or std <= 0:
            return None

        return self._cap_score(abs(value - good_value) / std)

    @staticmethod
    def _mean_valid(values):
        valid_values = []

        for value in values:
            if value is not None and not pd.isna(value):
                valid_values.append(value)

        if not valid_values:
            return 0.0

        return float(np.mean(valid_values))


    @staticmethod
    def _find_column(df, possible_names):
        for name in possible_names:
            if name in df.columns:
                return name

        return None

    @staticmethod
    def _cap_score(score, max_score=3.0):
        if score is None or pd.isna(score):
            return None
        return min(float(score), max_score)
