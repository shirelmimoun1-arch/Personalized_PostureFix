# ============================================================
# MODEL-AWARE PERSONALIZED POSTURE FEEDBACK
# ============================================================
#
# Purpose:
#   Generate interpretable and personalized corrective feedback
#   during real-time posture monitoring.
#
#   The feedback system contains two complementary components:
#
#   1. Model-aware explanation:
#      For frames classified as Bad, analyze the local geometry of
#      the personalized KNN classifier and identify the posture
#      characteristics that provide the strongest local evidence
#      for the prediction.
#
#   2. Supplementary geometric diagnostics:
#      Evaluate selected interpretable posture measurements that are
#      not part of the final classifier but can still provide useful
#      corrective guidance. These diagnostics are explicitly kept
#      separate from the explanation of the KNN decision.
#
# Main idea:
#   1. Reproduce the exact preprocessing used by the personalized model.
#   2. Verify the current KNN prediction and its Bad-class probability.
#   3. Inspect nearby Good and Bad calibration samples in the same
#      standardized feature space used by the classifier.
#   4. Decompose local Good-vs-Bad distance differences feature-by-feature.
#   5. Build a local personalized counterfactual using nearby Good samples.
#   6. Aggregate model evidence into human-interpretable posture groups.
#   7. Generate the strongest supported model-aware correction.
#   8. Optionally add personalized supplementary diagnostics, such as
#      shoulder asymmetry, without modifying the classifier decision.
#
# Notes:
#   The class expects the model package saved by train_personal_model.py.
#   Supplementary diagnostics must not be interpreted as features that
#   caused or contributed to the KNN prediction unless those features
#   are explicitly present in the trained model's feature set.
# ============================================================

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
import joblib
import numpy as np
import pandas as pd


class PostureFeedback:
    """
    Generate personalized corrective posture feedback using local
    KNN explanations and supplementary geometric diagnostics.
    """

    FEATURE_GROUPS: Mapping[str, Tuple[str, ...]] = {
        "head_forward": (
            "headForwardDepth",
        ),

        "head_vertical": (
            "headHeight",
            "earCenterHeightNorm",
            "eyeCenterHeightNorm",
            "leftEar_y",
            "rightEye_y",
        ),

        "head_orientation": (
            "eyeWidth",
        ),
    }

    GENERIC_BAD_MESSAGE = (
        "Your posture differs from your personalized Good-posture reference. "
        "Try returning to your comfortable calibrated upright position."
    )

    BORDERLINE_BAD_MESSAGE = (
        "Your posture is close to the model's decision boundary. "
        "Try returning gently toward your calibrated Good posture."
    )

    def __init__(
            self,
            model_pack_or_path,
            *,
            calibration_dataset_path=None,
            local_reference_k=None,
            min_bad_probability=0.50,
            min_group_evidence=0.08,
            global_feature_importance=None,
            shoulder_asymmetry_z_threshold=2.5,
    ):
        """
        Initialize the model-aware feedback engine.

        Args:
            model_pack_or_path:
                Model-package dictionary produced by train_personal_model.py
                or a path to the corresponding joblib file.
            calibration_dataset_path:
                Optional path to the user's calibration feature dataset.
                When provided, it is used to construct personalized
                references for supplementary geometric diagnostics.
                The dataset does not alter the fitted KNN classifier.
            local_reference_k:
                Number of nearest samples from each class used to estimate
                local Good and Bad reference regions. Defaults to model K.
            min_bad_probability:
                Minimum Bad probability required for a specific explanation.
            min_group_evidence:
                Minimum normalized evidence required for a semantic group.
            global_feature_importance:
                Optional mapping of global feature importances. When supplied,
                local evidence is softly weighted by these values.
            shoulder_asymmetry_z_threshold:
                Minimum robust standardized deviation from the user's
                calibrated Good shoulder alignment required to activate
                supplementary shoulder-asymmetry feedback.
        """
        if isinstance(model_pack_or_path, (str, Path)):
            self.model_pack = joblib.load(model_pack_or_path)
        else:
            self.model_pack = dict(model_pack_or_path)

        required_keys = {"imputer", "scaler", "model", "feature_names"}
        missing_keys = required_keys - set(self.model_pack.keys())
        if missing_keys:
            raise ValueError(
                "Model package is missing required keys: "
                + ", ".join(sorted(missing_keys))
            )

        self.imputer = self.model_pack["imputer"]
        self.scaler = self.model_pack["scaler"]
        self.model = self.model_pack["model"]
        self.feature_names = list(self.model_pack["feature_names"])

        if not self.feature_names:
            raise ValueError("Model package contains no feature names")

        if not hasattr(self.model, "_fit_X") or not hasattr(self.model, "_y"):
            raise ValueError("The KNN model must be fitted before feedback is created")

        self.train_X_scaled = np.asarray(self.model._fit_X, dtype=float)
        self.train_y = np.asarray(self.model._y).reshape(-1)

        if self.train_X_scaled.ndim != 2:
            raise ValueError("Unexpected KNN training matrix shape")

        if self.train_X_scaled.shape[1] != len(self.feature_names):
            raise ValueError(
                "Feature-name count does not match the fitted KNN feature dimension"
            )

        self.bad_class, self.good_class = self._resolve_numeric_classes()

        model_k = int(getattr(self.model, "n_neighbors", 5))
        if local_reference_k is None:
            local_reference_k = model_k

        self.local_reference_k = max(1, int(local_reference_k))
        self.min_bad_probability = float(min_bad_probability)
        self.min_group_evidence = float(min_group_evidence)

        if not 0.0 <= self.min_bad_probability <= 1.0:
            raise ValueError("min_bad_probability must be in [0, 1]")
        if self.min_group_evidence < 0.0:
            raise ValueError("min_group_evidence must be non-negative")

        self.global_feature_importance = self._prepare_global_importance(
            global_feature_importance
        )

        self.good_indices = np.flatnonzero(self.train_y == self.good_class)
        self.bad_indices = np.flatnonzero(self.train_y == self.bad_class)

        if len(self.good_indices) == 0:
            raise ValueError("The fitted KNN contains no Good training samples")
        if len(self.bad_indices) == 0:
            raise ValueError("The fitted KNN contains no Bad training samples")

        self.metric = self._resolve_metric()
        self.metric_p = self._resolve_metric_p()

        # ------------------------------------------------------------
        # Supplementary personalized shoulder diagnostic
        # ------------------------------------------------------------

        self.shoulder_asymmetry_z_threshold = float(
            shoulder_asymmetry_z_threshold
        )

        self.good_shoulder_median = None
        self.good_shoulder_scale = None

        if calibration_dataset_path is not None:
            self._initialize_shoulder_reference(
                calibration_dataset_path
            )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def explain(self, current_features: Mapping[str, float], top_k: int = 1) -> str:
        """
        Return personalized corrective feedback for one current frame.

        For frames classified as Bad, supplementary geometric diagnostics
        are evaluated first. If no sufficiently strong supplementary
        deviation is detected, feedback is derived from the local geometry
        of the personalized KNN model.

        Supplementary diagnostics do not participate in classification and
        must not be interpreted as explanations of the KNN prediction.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        details = self.explain_details(current_features)

        if details["prediction"] != "Bad":
            return ""

        shoulder_message = self._shoulder_asymmetry_message(
            current_features
        )

        if shoulder_message is not None:
            return shoulder_message

        if details["bad_probability"] < self.min_bad_probability:
            return self.BORDERLINE_BAD_MESSAGE

        messages: List[str] = []

        for group in details["ranked_groups"]:
            if group["evidence"] < self.min_group_evidence:
                continue

            message = self._message_for_group(
                group_name=group["group"],
                current_raw=details["current_raw"],
                good_target_raw=details["good_target_raw"],
                feature_evidence=details["feature_evidence"],
            )

            if message and message not in messages:
                messages.append(message)

            if len(messages) >= top_k:
                break

        if not messages:
            return self.GENERIC_BAD_MESSAGE

        return " ".join(messages)

    def explain_details(self, current_features: Mapping[str, float]) -> Dict:
        """
        Return a structured local explanation for debugging and evaluation.

        The returned dictionary contains model prediction, Bad probability,
        actual KNN neighbors, class-specific local references, feature-level
        evidence, a local Good counterfactual, and semantic group ranking.
        """
        current_imputed, current_scaled = self._prepare_current_vector(
            current_features
        )

        pred_numeric = self.model.predict(current_scaled)[0]
        prediction = self._numeric_to_text_label(pred_numeric)
        bad_probability = self._bad_probability(current_scaled)

        actual_neighbor_distances, actual_neighbor_indices = self.model.kneighbors(
            current_scaled,
            n_neighbors=min(
                int(getattr(self.model, "n_neighbors", self.local_reference_k)),
                len(self.train_X_scaled),
            ),
            return_distance=True,
        )

        actual_neighbor_distances = actual_neighbor_distances[0]
        actual_neighbor_indices = actual_neighbor_indices[0]
        actual_neighbor_labels = [
            self._numeric_to_text_label(self.train_y[idx])
            for idx in actual_neighbor_indices
        ]

        all_feature_distances = self._feature_distance_matrix(
            current_scaled[0],
            self.train_X_scaled,
        )
        all_total_distances = self._total_distance_from_feature_distances(
            all_feature_distances
        )

        good_local_indices = self._nearest_class_indices(
            all_total_distances,
            self.good_indices,
            self.local_reference_k,
        )
        bad_local_indices = self._nearest_class_indices(
            all_total_distances,
            self.bad_indices,
            self.local_reference_k,
        )

        good_feature_distance = self._mean_feature_distance(
            all_feature_distances,
            all_total_distances,
            good_local_indices,
        )
        bad_feature_distance = self._mean_feature_distance(
            all_feature_distances,
            all_total_distances,
            bad_local_indices,
        )

        # Positive evidence means that a feature is locally farther from Good
        # than from Bad, and therefore supports the Bad prediction.
        raw_local_evidence = np.maximum(
            good_feature_distance - bad_feature_distance,
            0.0,
        )
        weighted_local_evidence = (
            raw_local_evidence * self.global_feature_importance
        )

        evidence_sum = float(np.sum(weighted_local_evidence))
        if evidence_sum > 0:
            normalized_evidence = weighted_local_evidence / evidence_sum
        else:
            normalized_evidence = np.zeros_like(weighted_local_evidence)

        good_target_scaled = self._robust_good_counterfactual(
            good_local_indices
        )
        good_target_imputed = self.scaler.inverse_transform(
            good_target_scaled.reshape(1, -1)
        )[0]

        feature_evidence = {
            name: float(normalized_evidence[i])
            for i, name in enumerate(self.feature_names)
        }
        current_raw_dict = {
            name: float(current_imputed[0, i])
            for i, name in enumerate(self.feature_names)
        }
        good_target_raw_dict = {
            name: float(good_target_imputed[i])
            for i, name in enumerate(self.feature_names)
        }

        ranked_groups = self._rank_semantic_groups(feature_evidence)

        return {
            "prediction": prediction,
            "predicted_numeric_class": self._python_scalar(pred_numeric),
            "bad_probability": float(bad_probability),
            "actual_neighbor_indices": [int(x) for x in actual_neighbor_indices],
            "actual_neighbor_distances": [float(x) for x in actual_neighbor_distances],
            "actual_neighbor_labels": actual_neighbor_labels,
            "good_local_indices": [int(x) for x in good_local_indices],
            "bad_local_indices": [int(x) for x in bad_local_indices],
            "feature_evidence": feature_evidence,
            "ranked_features": sorted(
                feature_evidence.items(),
                key=lambda item: item[1],
                reverse=True,
            ),
            "ranked_groups": ranked_groups,
            "current_raw": current_raw_dict,
            "good_target_raw": good_target_raw_dict,
        }

    # ============================================================
    # MODEL / LABEL HANDLING
    # ============================================================

    def _resolve_numeric_classes(self):
        """Resolve numeric Good and Bad labels stored by the model."""
        label_to_int = self.model_pack.get("label_to_int")

        if isinstance(label_to_int, Mapping):
            normalized = {
                str(key).strip().lower(): value
                for key, value in label_to_int.items()
            }
            if "bad" in normalized and "good" in normalized:
                return normalized["bad"], normalized["good"]

        int_to_label = self.model_pack.get("int_to_label")

        if isinstance(int_to_label, Mapping):
            bad_class = None
            good_class = None

            for numeric, text in int_to_label.items():
                label = str(text).strip().lower()
                if label == "bad":
                    bad_class = numeric
                elif label == "good":
                    good_class = numeric

            if bad_class is not None and good_class is not None:
                return bad_class, good_class

        classes = list(getattr(self.model, "classes_", []))
        if 0 in classes and 1 in classes:
            return 0, 1

        raise ValueError(
            "Could not identify numeric Bad and Good classes from model package"
        )

    def _numeric_to_text_label(self, numeric_label) -> str:
        if numeric_label == self.bad_class:
            return "Bad"
        if numeric_label == self.good_class:
            return "Good"
        return "Unknown"

    def _bad_probability(self, current_scaled: np.ndarray) -> float:
        if not hasattr(self.model, "predict_proba"):
            prediction = self.model.predict(current_scaled)[0]
            return 1.0 if prediction == self.bad_class else 0.0

        probabilities = self.model.predict_proba(current_scaled)[0]
        classes = list(self.model.classes_)

        try:
            bad_position = classes.index(self.bad_class)
        except ValueError:
            return 0.0

        return float(probabilities[bad_position])

    # ============================================================
    # PREPROCESSING
    # ============================================================

    def _prepare_current_vector(
        self,
        current_features: Mapping[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reproduce feature ordering, imputation, and scaling from training."""
        row = {
            name: current_features.get(name, np.nan)
            for name in self.feature_names
        }
        frame = pd.DataFrame([row], columns=self.feature_names)

        current_imputed = self.imputer.transform(frame)
        current_scaled = self.scaler.transform(current_imputed)

        return (
            np.asarray(current_imputed, dtype=float),
            np.asarray(current_scaled, dtype=float),
        )

    # ============================================================
    # LOCAL KNN EXPLANATION
    # ============================================================

    def _resolve_metric(self) -> str:
        metric = getattr(self.model, "effective_metric_", None)
        if metric is None:
            metric = getattr(self.model, "metric", "minkowski")
        return str(metric).lower()

    def _resolve_metric_p(self) -> float:
        params = getattr(self.model, "effective_metric_params_", {}) or {}
        if "p" in params:
            return float(params["p"])
        return float(getattr(self.model, "p", 2))

    def _feature_distance_matrix(
        self,
        current_scaled: np.ndarray,
        reference_scaled: np.ndarray,
    ) -> np.ndarray:
        """Compute feature-wise distance contributions in model space."""
        delta = reference_scaled - current_scaled.reshape(1, -1)
        abs_delta = np.abs(delta)

        if self.metric in {"manhattan", "cityblock", "l1"}:
            return abs_delta
        if self.metric in {"euclidean", "l2"}:
            return np.square(abs_delta)
        if self.metric == "minkowski":
            return np.power(abs_delta, self.metric_p)

        # Conservative fallback for uncommon metrics.
        return abs_delta

    def _total_distance_from_feature_distances(
        self,
        feature_distances: np.ndarray,
    ) -> np.ndarray:
        summed = np.sum(feature_distances, axis=1)

        if self.metric in {"euclidean", "l2"}:
            return np.sqrt(summed)
        if self.metric == "minkowski":
            return np.power(summed, 1.0 / self.metric_p)
        return summed

    @staticmethod
    def _nearest_class_indices(
        all_distances: np.ndarray,
        class_indices: np.ndarray,
        k: int,
    ) -> np.ndarray:
        k = min(int(k), len(class_indices))
        class_distances = all_distances[class_indices]
        local_order = np.argsort(class_distances)[:k]
        return class_indices[local_order]

    @staticmethod
    def _inverse_distance_weights(
        distances: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        distances = np.asarray(distances, dtype=float)
        weights = 1.0 / np.maximum(distances, eps)
        weight_sum = float(np.sum(weights))

        if weight_sum <= 0 or not np.isfinite(weight_sum):
            return np.full(
                shape=len(distances),
                fill_value=1.0 / len(distances),
                dtype=float,
            )

        return weights / weight_sum

    def _mean_feature_distance(
        self,
        feature_distances: np.ndarray,
        total_distances: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Estimate local per-feature distance to one class region."""
        selected_total = total_distances[indices]
        selected_features = feature_distances[indices]
        weights = self._inverse_distance_weights(selected_total)

        return np.average(
            selected_features,
            axis=0,
            weights=weights,
        )

    def _robust_good_counterfactual(
        self,
        good_local_indices: np.ndarray,
    ) -> np.ndarray:
        """Use the coordinate-wise median of nearby Good samples as target."""
        local_good = self.train_X_scaled[good_local_indices]
        return np.median(local_good, axis=0)

    # ============================================================
    # EVIDENCE AGGREGATION
    # ============================================================

    def _prepare_global_importance(
        self,
        global_feature_importance: Optional[Mapping[str, float]],
    ) -> np.ndarray:
        if global_feature_importance is None:
            return np.ones(len(self.feature_names), dtype=float)

        raw = np.array(
            [
                max(0.0, float(global_feature_importance.get(name, 0.0)))
                for name in self.feature_names
            ],
            dtype=float,
        )

        if np.all(raw == 0):
            return np.ones(len(self.feature_names), dtype=float)

        normalized = raw / float(np.max(raw))

        # Soft weighting only: global importance must not dominate the local
        # explanation of this particular frame.
        return 0.5 + normalized

    def _rank_semantic_groups(
            self,
            feature_evidence: Mapping[str, float],
    ) -> List[Dict]:

        ranked_groups = []

        for group_name, group_features in self.FEATURE_GROUPS.items():
            evidence = float(
                sum(
                    feature_evidence.get(feature, 0.0)
                    for feature in group_features
                )
            )

            ranked_groups.append(
                {
                    "group": group_name,
                    "evidence": evidence,
                    "features": list(group_features),
                }
            )

        ranked_groups.sort(
            key=lambda item: item["evidence"],
            reverse=True,
        )

        return ranked_groups

    # ============================================================
    # SUPPLEMENTARY GEOMETRIC DIAGNOSTICS
    # ============================================================

    def _initialize_shoulder_reference(self, calibration_dataset_path):
        """
        Build a personalized Good-posture reference for shoulder symmetry.

        shoulderTilt is intentionally used only as a supplementary
        diagnostic feature and does not participate in KNN classification.

        The reference is estimated exclusively from the user's calibrated
        Good-posture samples. A robust median and median absolute deviation
        (MAD) are used to reduce sensitivity to noisy pose landmarks.
        """

        path = Path(calibration_dataset_path)

        if not path.exists():
            print(
                "Warning: calibration dataset was not found; "
                "shoulder feedback will be disabled."
            )
            return

        if path.suffix.lower() in {".xlsx", ".xls"}:
            calibration_df = pd.read_excel(path)

        elif path.suffix.lower() == ".csv":
            calibration_df = pd.read_csv(path)

        else:
            print(
                "Warning: unsupported calibration dataset format; "
                "shoulder feedback will be disabled."
            )
            return

        required_columns = {"Label", "shoulderTilt"}

        if not required_columns.issubset(calibration_df.columns):
            print(
                "Warning: calibration dataset does not contain "
                "Label and shoulderTilt; shoulder feedback disabled."
            )
            return

        labels = (
            calibration_df["Label"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        good_values = pd.to_numeric(
            calibration_df.loc[
                labels == "good",
                "shoulderTilt"
            ],
            errors="coerce"
        ).dropna()

        if len(good_values) < 10:
            print(
                "Warning: insufficient Good shoulderTilt samples; "
                "shoulder feedback disabled."
            )
            return

        values = good_values.to_numpy(dtype=float)

        median = float(np.median(values))

        mad = float(
            np.median(
                np.abs(values - median)
            )
        )

        # Convert MAD to a robust estimate comparable to standard deviation.
        robust_scale = 1.4826 * mad

        # In very stable calibration data MAD can approach zero.
        # Fall back to ordinary standard deviation in that case.
        if robust_scale < 1e-6:
            robust_scale = float(np.std(values))

        if robust_scale < 1e-6:
            print(
                "Warning: shoulderTilt variation in Good calibration "
                "is too small to construct a reliable diagnostic."
            )
            return

        self.good_shoulder_median = median
        self.good_shoulder_scale = robust_scale

    def _shoulder_asymmetry_message(
            self,
            current_features: Mapping[str, float],
    ) -> Optional[str]:
        """
        Return supplementary feedback when shoulder alignment deviates
        strongly from the user's personalized Good-posture reference.

        This diagnostic is independent of the KNN decision and therefore
        must not be interpreted as an explanation of the classifier output.
        """

        if (
                self.good_shoulder_median is None
                or self.good_shoulder_scale is None
        ):
            return None

        shoulder_tilt = current_features.get(
            "shoulderTilt",
            np.nan
        )

        try:
            shoulder_tilt = float(shoulder_tilt)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(shoulder_tilt):
            return None

        deviation = abs(
            shoulder_tilt - self.good_shoulder_median
        )

        robust_z = (
                deviation / self.good_shoulder_scale
        )

        if robust_z < self.shoulder_asymmetry_z_threshold:
            return None

        return (
            "Your shoulders appear less level than in your calibrated "
            "Good posture. Try leveling and relaxing your shoulders."
        )


    # ============================================================
    # SEMANTIC COUNTERFACTUAL MESSAGES
    # ============================================================

    def _message_for_group(
            self,
            group_name: str,
            current_raw: Mapping[str, float],
            good_target_raw: Mapping[str, float],
            feature_evidence: Mapping[str, float],
    ) -> Optional[str]:

        if group_name == "head_forward":
            return self._head_forward_message(
                current_raw,
                good_target_raw,
            )

        if group_name == "head_vertical":
            return self._head_vertical_message(
                current_raw,
                good_target_raw,
                feature_evidence,
            )

        if group_name == "head_orientation":
            return self._head_orientation_message()

        return None

    @staticmethod
    def _head_forward_message(
        current_raw: Mapping[str, float],
        good_target_raw: Mapping[str, float],
    ) -> str:
        feature = "headForwardDepth"

        if feature not in current_raw or feature not in good_target_raw:
            return (
                "Your forward head position differs from your calibrated Good posture. "
                "Try returning your head toward your comfortable neutral position."
            )

        current = current_raw[feature]
        target = good_target_raw[feature]
        correction = target - current

        if correction < 0:
            return (
                "Your head appears farther forward than in your nearby Good-posture "
                "examples. Try moving your head slightly backward while keeping your "
                "gaze comfortable."
            )

        return (
            "Your head depth differs from your nearby Good-posture examples. "
            "Try returning your head gently toward your calibrated neutral position."
        )

    @staticmethod
    def _head_vertical_message(
        current_raw: Mapping[str, float],
        good_target_raw: Mapping[str, float],
        feature_evidence: Mapping[str, float],
    ) -> str:
        # Prefer headHeight because its direction is geometrically clearer
        # than raw image y-coordinates.
        if (
            "headHeight" in current_raw
            and "headHeight" in good_target_raw
            and feature_evidence.get("headHeight", 0.0) > 0
        ):
            current = current_raw["headHeight"]
            target = good_target_raw["headHeight"]

            if current < target:
                return (
                    "Your head and upper body appear lower than in your Good-posture examples. "
                    "You may be slouching or rounding your back. Try sitting taller and "
                    "straightening your upper back."
                )

            if current > target:
                return (
                    "Your head is higher than in your nearby Good-posture examples. "
                    "Try relaxing your neck and returning your gaze toward your "
                    "calibrated neutral level."
                )

        # Raw vertical coordinates are camera-dependent, so the fallback is
        # intentionally conservative.
        return (
            "Your vertical head position differs from your nearby calibrated "
            "Good-posture examples. Try returning your head and upper body toward "
            "your comfortable upright position."
        )

    @staticmethod
    def _head_orientation_message() -> str:
        return (
            "Your facial geometry differs from your nearby Good-posture examples. "
            "Try facing the screen directly and returning to your calibrated "
            "head position."
        )

    # ============================================================
    # SMALL UTILITIES
    # ============================================================

    @staticmethod
    def _python_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        return value
