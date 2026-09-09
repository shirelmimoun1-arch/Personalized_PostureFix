# ============================================================
# PERSONALIZED KNN TRAINING AND EVALUATION PIPELINE
# ============================================================
#
# Purpose
# -------
# Train, evaluate, and serialize a personalized KNN posture classifier
# using user-specific calibration data.
#
# The pipeline is designed to estimate generalization to unseen posture
# segments while preventing frames originating from the same recorded
# segment from being shared between training and evaluation partitions.
#
#
# Methodological overview
# -----------------------
# 1. Load the user's calibration dataset and retain labeled Good and Bad
#    posture samples.
#
# 2. Partition complete posture segments into:
#       - a Train/CV pool,
#       - a held-out test set.
#
#    The split is performed independently within each posture class so
#    that both Good and Bad segments are represented in the held-out set.
#
# 3. Restrict the model input to the final shared personalized feature
#    representation selected previously using Train/CV data only.
#
# 4. Select KNN hyperparameters using StratifiedGroupKFold on the
#    Train/CV pool. Segment_ID is used as the grouping variable so that
#    frames from the same recorded posture segment remain within a
#    single cross-validation partition.
#
# 5. Within every cross-validation fold:
#       - fit median imputation on the fold-training data,
#       - fit standardization on the fold-training data,
#       - train the candidate KNN,
#       - evaluate on the corresponding validation fold.
#
# 6. Rank KNN configurations according to:
#       - highest mean balanced accuracy,
#       - then highest mean Bad-class F1,
#       - then lower cross-validation accuracy variability.
#
# 7. Fit an evaluation model on the complete Train/CV pool using the
#    selected hyperparameters and evaluate it once on the held-out
#    posture segments.
#
# 8. Report Accuracy, Balanced Accuracy, Bad-class Recall, Bad-class F1,
#    the complete classification report, and the confusion matrix.
#
# 9. Optionally perform qualitative error analysis on held-out
#    predictions by inspecting Bad-posture samples incorrectly
#    classified as Good. This analysis does not modify the model.
#
# 10. After evaluation is complete, refit the imputer, scaler, and KNN
#     on the complete calibration dataset using the already selected
#     hyperparameters. This final model is serialized for live inference.
#
#
# Leakage prevention
# ------------------
# Model-selection decisions are made without access to the held-out test
# segments. In particular, the test set is not used for:
#   - feature selection,
#   - KNN hyperparameter selection,
#   - fitting cross-validation imputers,
#   - fitting cross-validation scalers,
#   - training the evaluation classifier.
#
# Within cross-validation, all preprocessing operations are fitted only
# on the training portion of each fold.
#
# The final refit on the complete calibration dataset occurs only after
# held-out evaluation and does not alter the previously reported test
# metrics or selected hyperparameters.
#
#
# Final personalized feature representation
# -----------------------------------------
# The classifier uses a fixed seven-feature representation selected by
# backward sequential feature elimination across personalized users:
#
#   - earCenterHeightNorm
#   - eyeCenterHeightNorm
#   - eyeWidth
#   - headForwardDepth
#   - headHeight
#   - leftEar_y
#   - rightEye_y
#
#
# Saved model package
# -------------------
# The serialized model package contains:
#   - fitted median imputer,
#   - fitted StandardScaler,
#   - fitted KNN classifier,
#   - ordered feature names,
#   - class mappings,
#   - selected KNN hyperparameters,
#   - held-out evaluation metrics,
#   - cross-validation search results.
#
# ============================================================

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shutil

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score
)
from sklearn.impute import SimpleImputer

LABEL_TO_INT = {"Bad": 0, "Good": 1}
INT_TO_LABEL = {0: "Bad", 1: "Good"}

PERSONAL_SELECTED_FEATURES = [
    "earCenterHeightNorm",
    "eyeCenterHeightNorm",
    "eyeWidth",
    "headForwardDepth",
    "headHeight",
    "leftEar_y",
    "rightEye_y",
]

def split_personal_dataset_by_segment_train_test(
    df,
    test_segment_ratio=0.25,
    random_state=42
):
    """
    Split a personalized calibration dataset into Train/CV and held-out
    test partitions at the posture-segment level.

    Complete Segment_ID groups, rather than individual frames, are assigned
    to either Train/CV or test. The procedure is performed independently
    for each posture class, ensuring that at least one segment from every
    class is reserved for held-out evaluation.

    Segment-level splitting prevents temporally related frames originating
    from the same recorded posture instance from appearing in both model
    development and final evaluation.

    Parameters
    ----------
    df : pandas.DataFrame
        Personalized calibration dataset containing Label and Segment_ID.
    test_segment_ratio : float, default=0.25
        Approximate fraction of segments from each class assigned to the
        held-out test partition.
    random_state : int, default=42
        Seed controlling reproducible segment shuffling.

    Returns
    -------
    train_cv_df : pandas.DataFrame
        Samples belonging to segments reserved for model development and
        grouped cross-validation.
    test_df : pandas.DataFrame
        Samples belonging to held-out posture segments used only for final
        evaluation and subsequent post-hoc analysis.

    Raises
    ------
    RuntimeError
        If Segment_ID is unavailable or a posture class contains too few
        segments to construct separate Train/CV and test partitions.
    """
    if "Segment_ID" not in df.columns:
        raise RuntimeError(
            "Segment-based train/test split requires a 'Segment_ID' column."
        )

    test_indices = []
    rng = np.random.default_rng(random_state)

    for label in df["Label"].unique():
        label_df = df[df["Label"] == label]

        segment_ids = label_df["Segment_ID"].dropna().unique().tolist()

        if len(segment_ids) < 2:
            raise RuntimeError(
                f"Need at least 2 segments for label '{label}' to create "
                "a train/CV pool and a test set."
            )

        rng.shuffle(segment_ids)

        num_test_segments = max(
            1,
            int(len(segment_ids) * test_segment_ratio)
        )

        if num_test_segments >= len(segment_ids):
            raise RuntimeError(
                f"Not enough segments for label '{label}'."
            )

        test_segments = segment_ids[:num_test_segments]

        train_segments = [
            seg for seg in segment_ids
            if seg not in test_segments
        ]

        print(f"\nLabel: {label}")
        print(f"All segments: {sorted(segment_ids)}")
        print(f"Test segments: {sorted(test_segments)}")
        print(f"Train/CV segments: {sorted(train_segments)}")

        label_test_indices = label_df[
            label_df["Segment_ID"].isin(test_segments)
        ].index.tolist()

        test_indices.extend(label_test_indices)

    test_indices = set(test_indices)

    test_df = df.loc[list(test_indices)].copy()
    train_cv_df = df.drop(list(test_indices)).copy()

    return train_cv_df, test_df


def build_segment_summary(split_dfs):
    """
    Generate a textual summary of dataset partitions.

    For each supplied partition, the summary reports the total number
    of samples, the class distribution, and the number of unique posture
    segments associated with each class.

    Parameters
    ----------
    split_dfs : iterable of tuple
        Sequence of ``(split_name, dataframe)`` pairs describing the
        dataset partitions to summarize.

    Returns
    -------
    str
        Human-readable summary of sample counts, label distributions,
        and segment counts for each supplied partition.
    """
    lines = []

    for split_name, split_df in split_dfs:
        lines.append(f"{split_name} samples: {len(split_df)}")
        lines.append(f"{split_name} labels:")
        lines.append(split_df["Label"].value_counts().to_string())

        if "Segment_ID" in split_df.columns:
            lines.append(f"{split_name} segments:")
            lines.append(
                split_df.groupby("Label")["Segment_ID"]
                .nunique()
                .to_string()
            )

        lines.append("")

    return "\n".join(lines)


def search_best_knn_params_group_cv(
    X,
    y,
    groups,
    k_values=None,
    max_cv_splits=5
):
    """
    Select the best KNN hyperparameters using
    segment-grouped stratified cross-validation.

    Selection order:
    1. Highest mean Balanced Accuracy
    2. Highest mean Bad-class F1
    3. Lowest std of Accuracy
    """

    if k_values is None:
        k_values = [
            3, 5, 7, 9, 11,
            15, 21, 31, 41, 51
        ]

    unique_groups = (
        pd.Series(groups)
        .dropna()
        .unique()
    )

    num_groups = len(unique_groups)

    if num_groups < 2:
        raise RuntimeError(
            "StratifiedGroupKFold requires at least "
            "2 unique Segment_ID groups."
        )

    n_splits = min(
        max_cv_splits,
        num_groups
    )

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    results = []

    for k in k_values:
        for weights in ["uniform", "distance"]:
            for metric in ["euclidean", "manhattan"]:

                fold_accuracies = []
                fold_balanced_accuracies = []
                fold_bad_recalls = []
                fold_bad_f1s = []

                skipped_folds = 0

                for train_idx, val_idx in cv.split(
                    X,
                    y,
                    groups=groups
                ):
                    X_fold_train = X.iloc[train_idx]
                    y_fold_train = y.iloc[train_idx]

                    X_fold_val = X.iloc[val_idx]
                    y_fold_val = y.iloc[val_idx]

                    if k > len(y_fold_train):
                        skipped_folds += 1
                        continue

                    # ----------------------------------------
                    # Fold-local imputation
                    # ----------------------------------------

                    imputer = SimpleImputer(
                        strategy="median"
                    )

                    X_fold_train_imputed = (
                        imputer.fit_transform(
                            X_fold_train
                        )
                    )

                    X_fold_val_imputed = (
                        imputer.transform(
                            X_fold_val
                        )
                    )

                    # ----------------------------------------
                    # Fold-local scaling
                    # ----------------------------------------

                    scaler = StandardScaler()

                    X_fold_train_scaled = (
                        scaler.fit_transform(
                            X_fold_train_imputed
                        )
                    )

                    X_fold_val_scaled = (
                        scaler.transform(
                            X_fold_val_imputed
                        )
                    )

                    # ----------------------------------------
                    # KNN
                    # ----------------------------------------

                    candidate_model = KNeighborsClassifier(
                        n_neighbors=k,
                        weights=weights,
                        metric=metric
                    )

                    candidate_model.fit(
                        X_fold_train_scaled,
                        y_fold_train
                    )

                    fold_pred = candidate_model.predict(
                        X_fold_val_scaled
                    )

                    # ----------------------------------------
                    # Metrics
                    # Bad = 0
                    # Good = 1
                    # ----------------------------------------

                    fold_accuracies.append(
                        accuracy_score(
                            y_fold_val,
                            fold_pred
                        )
                    )

                    fold_balanced_accuracies.append(
                        balanced_accuracy_score(
                            y_fold_val,
                            fold_pred
                        )
                    )

                    fold_bad_recalls.append(
                        recall_score(
                            y_fold_val,
                            fold_pred,
                            pos_label=LABEL_TO_INT["Bad"],
                            zero_division=0
                        )
                    )

                    fold_bad_f1s.append(
                        f1_score(
                            y_fold_val,
                            fold_pred,
                            pos_label=LABEL_TO_INT["Bad"],
                            zero_division=0
                        )
                    )

                if not fold_accuracies:
                    continue

                results.append({
                    "n_neighbors": k,
                    "weights": weights,
                    "metric": metric,

                    "mean_cv_accuracy":
                        float(
                            np.mean(
                                fold_accuracies
                            )
                        ),

                    "std_cv_accuracy":
                        float(
                            np.std(
                                fold_accuracies
                            )
                        ),

                    "mean_cv_balanced_accuracy":
                        float(
                            np.mean(
                                fold_balanced_accuracies
                            )
                        ),

                    "mean_cv_bad_recall":
                        float(
                            np.mean(
                                fold_bad_recalls
                            )
                        ),

                    "mean_cv_bad_f1":
                        float(
                            np.mean(
                                fold_bad_f1s
                            )
                        ),

                    "num_folds_used":
                        len(fold_accuracies),

                    "num_folds_skipped":
                        skipped_folds
                })

    if not results:
        raise RuntimeError(
            "No valid KNN hyperparameter configurations "
            "were evaluated."
        )

    cv_results_df = pd.DataFrame(results)

    # Same ranking logic used during feature selection
    cv_results_df = cv_results_df.sort_values(
        by=[
            "mean_cv_balanced_accuracy",
            "mean_cv_bad_f1",
            "std_cv_accuracy"
        ],
        ascending=[
            False,
            False,
            True
        ]
    ).reset_index(drop=True)

    best_row = cv_results_df.iloc[0]

    best_params = {
        "n_neighbors":
            int(
                best_row["n_neighbors"]
            ),

        "weights":
            best_row["weights"],

        "metric":
            best_row["metric"]
    }

    return best_params, cv_results_df


def save_k_cv_search_figure(cv_results_df, evaluation_dir):
    """
    Save a visualization of grouped cross-validation performance across
    KNN hyperparameter configurations.

    Mean balanced accuracy is plotted as a function of K separately for
    each distance-metric and neighbor-weighting combination.

    Parameters
    ----------
    cv_results_df : pandas.DataFrame
        Results produced by the grouped KNN hyperparameter search.
    evaluation_dir : str or Path
        Root directory in which the figure is saved.
    """
    hyper_dir = os.path.join(evaluation_dir, "hyperparameter_selection")
    os.makedirs(hyper_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = cv_results_df.groupby(["metric", "weights"])

    for (metric, weights), group_df in grouped:
        group_df = group_df.sort_values("n_neighbors")
        label = f"{metric} + {weights}"

        ax.plot(
            group_df["n_neighbors"],
            group_df["mean_cv_balanced_accuracy"],
            marker="o",
            label=label
        )

    ax.set_title(
        "K vs Mean Segment-Grouped CV Balanced Accuracy"
    )
    ax.set_xlabel("Number of Neighbors (K)")
    ax.set_ylabel(
        "Mean Balanced Accuracy"
    )
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    figure_path = os.path.join(
        hyper_dir,
        "K_vs_group_cv_balanced_accuracy.png"
    )
    plt.savefig(figure_path)
    plt.close()

    print(f"\nK-selection Group-CV figure saved to: {figure_path}")


def save_bad_predicted_good_mistakes(
    evaluation_dir,
    test_df,
    y_test,
    y_pred,
    frames_dir
):
    """
    Save held-out Bad-posture samples that were incorrectly classified
    as Good for qualitative error analysis.

    False-negative samples are recorded in a CSV file and, when available,
    their corresponding source frames are copied into a dedicated
    evaluation directory.

    This procedure is diagnostic only and does not modify the trained
    model or influence model-selection decisions.

    Parameters
    ----------
    evaluation_dir : str or Path
        Root directory for evaluation outputs.
    test_df : pandas.DataFrame
        Held-out test dataframe.
    y_test : array-like
        Ground-truth textual labels.
    y_pred : array-like
        Predicted textual labels.
    frames_dir : str or Path
        Directory containing the original calibration frames.
    """
    mistakes_dir = os.path.join(
        evaluation_dir,
        "misclassified",
        "bad_predicted_good"
    )

    # If folder already exists -> delete it completely
    if os.path.exists(mistakes_dir):
        shutil.rmtree(mistakes_dir)

    os.makedirs(mistakes_dir, exist_ok=True)

    mistakes = []
    copied_count = 0

    for i, (true_label, pred_label) in enumerate(zip(list(y_test), list(y_pred))):

        if true_label == "Bad" and pred_label == "Good":
            row = test_df.iloc[i].copy()

            image_id = row["Image_ID"]
            image_path = os.path.join(frames_dir, image_id)

            row["True_Label"] = true_label
            row["Predicted_Label"] = pred_label
            row["Source_Frame_Path"] = image_path

            mistakes.append(row)

            if os.path.exists(image_path):
                dst_path = os.path.join(mistakes_dir, image_id)
                shutil.copy(image_path, dst_path)
                copied_count += 1
            else:
                print(f"Warning: frame not found: {image_path}")

    mistakes_df = pd.DataFrame(mistakes)

    csv_path = os.path.join(
        mistakes_dir,
        "misclassified_bad_predicted_good.csv"
    )

    mistakes_df.to_csv(csv_path, index=False)

    print(f"\nSaved {len(mistakes_df)} Bad→Good mistakes to CSV: {csv_path}")
    print(f"Copied {copied_count} misclassified frame images to: {mistakes_dir}")


def save_evaluation_results(
    evaluation_dir,
    y_test,
    y_pred,
    accuracy,
    balanced_accuracy,
    bad_recall,
    bad_f1,
    report_dict,
    cm,
    class_names,
    cv_results_df=None,
    best_params=None,
    split_summary=None
):
    """
    Persist quantitative and visual results from held-out model evaluation.

    The function saves the principal classification metrics, complete
    classification report, grouped cross-validation search results,
    train/test split summary, and confusion matrix.

    Parameters
    ----------
    evaluation_dir : str or Path
        Destination directory for evaluation artifacts.
    y_test : array-like
        Ground-truth labels for the held-out test set.
    y_pred : array-like
        Model predictions for the held-out test set.
    accuracy : float
        Held-out classification accuracy.
    balanced_accuracy : float
        Held-out balanced accuracy.
    bad_recall : float
        Recall for the Bad posture class.
    bad_f1 : float
        F1 score for the Bad posture class.
    report_dict : dict
        Dictionary representation of the classification report.
    cm : numpy.ndarray
        Confusion matrix.
    class_names : sequence of str
        Ordered class names used to label the confusion matrix.
    cv_results_df : pandas.DataFrame or None
        Optional grouped cross-validation hyperparameter-search results.
    best_params : dict or None
        Hyperparameters selected using grouped cross-validation.
    split_summary : str or None
        Optional textual description of the Train/CV and test partitions.
    """
    os.makedirs(evaluation_dir, exist_ok=True)

    report_text = classification_report(y_test, y_pred, zero_division=0)

    with open(os.path.join(evaluation_dir, "classification_report.txt"), "w") as f:
        f.write(
            f"Accuracy: {accuracy:.4f}\n"
        )

        f.write(
            f"Balanced Accuracy: "
            f"{balanced_accuracy:.4f}\n"
        )

        f.write(
            f"Bad Recall: "
            f"{bad_recall:.4f}\n"
        )

        f.write(
            f"Bad F1: "
            f"{bad_f1:.4f}\n\n"
        )

        if best_params is not None:
            f.write(
                f"Best parameters chosen by StratifiedGroupKFold CV: "
                f"{best_params}\n\n"
            )

        f.write(report_text)

    metrics_df = pd.DataFrame(report_dict).transpose()
    metrics_df.to_csv(os.path.join(evaluation_dir, "metrics_summary.csv"))

    if cv_results_df is not None:
        hyper_dir = os.path.join(evaluation_dir, "hyperparameter_selection")
        os.makedirs(hyper_dir, exist_ok=True)

        cv_results_df.to_csv(
            os.path.join(hyper_dir, "group_cv_search_results.csv"),
            index=False
        )

    if split_summary is not None:
        with open(os.path.join(evaluation_dir, "split_summary.txt"), "w") as f:
            f.write(split_summary)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.savefig(os.path.join(evaluation_dir, "confusion_matrix.png"))
    plt.close()


def train_personal_model(
    dataset_path,
    model_output_path,
    frames_dir,
    evaluation_dir=None,
    split_method="segment",
    test_segment_ratio=0.25,
    random_state=42,
    max_cv_splits=5
):
    """
    Train, evaluate, and serialize a personalized KNN posture classifier.

    The function implements the complete personalized modeling pipeline.
    Calibration data are first partitioned by posture segment into a
    Train/CV pool and an independent held-out test set. KNN hyperparameters
    are selected exclusively within the Train/CV pool using
    StratifiedGroupKFold.

    After hyperparameter selection, an evaluation model is fitted on the
    complete Train/CV partition and evaluated once on the held-out posture
    segments. The resulting metrics characterize generalization to posture
    segments that were not used during model development.

    Once evaluation is complete, the selected preprocessing operations and
    KNN configuration are refitted on the complete calibration dataset.
    This final refitted model is serialized for deployment in the real-time
    personalized posture-monitoring system.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the personalized calibration feature dataset.
    model_output_path : str or Path
        Destination path for the serialized model package.
    frames_dir : str or Path
        Directory containing calibration frames used for qualitative
        inspection of misclassified held-out samples.
    evaluation_dir : str or Path or None
        Optional destination for evaluation reports, figures, grouped
        cross-validation results, and qualitative error-analysis outputs.
    split_method : str, default="segment"
        Dataset partitioning strategy. This implementation requires
        segment-level splitting.
    test_segment_ratio : float, default=0.25
        Approximate fraction of posture segments from each class reserved
        for held-out evaluation.
    random_state : int, default=42
        Seed controlling reproducible segment assignment.
    max_cv_splits : int, default=5
        Maximum number of grouped cross-validation folds.

    Returns
    -------
    dict
        Serialized-model package containing the final fitted imputer,
        scaler, KNN classifier, feature representation, class mappings,
        selected hyperparameters, cross-validation results, and held-out
        evaluation metrics.

    Notes
    -----
    The held-out test set is not used for hyperparameter selection or
    fitting of the evaluation preprocessing pipeline.

    Misclassification inspection, when enabled, is performed only after
    the evaluation predictions have been obtained. It is a post-hoc
    diagnostic analysis and does not influence model-selection decisions.

    The final deployment model is refitted on all available calibration
    samples only after held-out evaluation has been completed.
    """
    if split_method != "segment":
        raise ValueError(
            "This StratifiedGroupKFold version expects split_method='segment'."
        )

    df = pd.read_excel(dataset_path)
    df.columns = df.columns.str.strip()

    df = df[df["Label"].isin(["Good", "Bad"])].copy()

    if df.empty:
        raise RuntimeError("Dataset is empty after filtering labels.")

    if len(df["Label"].unique()) < 2:
        raise RuntimeError("Training requires both Good and Bad samples.")

    if "Segment_ID" not in df.columns:
        raise RuntimeError(
            "GroupKFold training requires a 'Segment_ID' column."
        )

    train_cv_df, test_df = split_personal_dataset_by_segment_train_test(
        df,
        test_segment_ratio=test_segment_ratio,
        random_state=random_state
    )

    if train_cv_df.empty or test_df.empty:
        raise RuntimeError(
            "Train/CV pool or test split failed. Not enough calibration samples."
        )

    split_summary = (
        f"Hold-out split method: segment-based\n"
        f"Cross-validation method: StratifiedGroupKFold by Segment_ID\n"
        f"Test segment ratio: {test_segment_ratio}\n"
        f"Max CV splits: {max_cv_splits}\n\n"
        + build_segment_summary([
            ("Train/CV pool", train_cv_df),
            ("Test", test_df)
        ])
    )

    print("\nTrain/CV Pool + Test split:")
    print(split_summary)

    target_column = "Label"

    # ============================================================
    # FINAL PERSONALIZED FEATURE REPRESENTATION
    #
    # Selected by backward sequential feature elimination
    # using Train/CV data only across all personalized users.
    # ============================================================

    feature_columns = PERSONAL_SELECTED_FEATURES.copy()

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Dataset is missing required personalized "
            f"model features: {missing_features}"
        )

    print(
        "\nFinal personalized feature representation "
        f"({len(feature_columns)} features):"
    )

    for feature in feature_columns:
        print(f"  - {feature}")

    X_train_cv = train_cv_df[feature_columns]
    y_train_cv_text = train_cv_df[target_column]
    y_train_cv = y_train_cv_text.map(LABEL_TO_INT)

    groups_train_cv = train_cv_df["Segment_ID"]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    feature_names = list(feature_columns)

    best_params, cv_results_df = search_best_knn_params_group_cv(
        X=X_train_cv,
        y=y_train_cv,
        groups=groups_train_cv,
        max_cv_splits=max_cv_splits
    )

    if evaluation_dir is not None:
        save_k_cv_search_figure(cv_results_df, evaluation_dir)

    print("\nBest parameters chosen by StratifiedGroupKFold CV:")
    print(best_params)

    print("\nTop Group-CV results:")
    print(cv_results_df.head())

    evaluation_imputer = SimpleImputer(strategy="median")
    X_train_cv_imputed = evaluation_imputer.fit_transform(X_train_cv)
    X_test_imputed = evaluation_imputer.transform(X_test)

    evaluation_scaler = StandardScaler()
    X_train_cv_scaled = evaluation_scaler.fit_transform(X_train_cv_imputed)
    X_test_scaled = evaluation_scaler.transform(X_test_imputed)

    evaluation_knn = KNeighborsClassifier(**best_params)
    evaluation_knn.fit(X_train_cv_scaled, y_train_cv)

    y_pred_numeric = evaluation_knn.predict(X_test_scaled)
    y_pred = [
        INT_TO_LABEL[int(label)]
        for label in y_pred_numeric
    ]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    bad_recall = recall_score(
        y_test,
        y_pred,
        pos_label="Bad",
        zero_division=0
    )

    bad_f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Bad",
        zero_division=0
    )

    class_names = [
        "Bad",
        "Good"
    ]

    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0
    )

    if evaluation_dir is not None:
        save_bad_predicted_good_mistakes(
            evaluation_dir=evaluation_dir,
            test_df=test_df,
            y_test=y_test,
            y_pred=y_pred,
            frames_dir=frames_dir
        )

    print(
        f"\nTest Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{balanced_accuracy:.2%}"
    )

    print(
        f"Test Bad Recall: "
        f"{bad_recall:.2%}"
    )

    print(
        f"Test Bad F1: "
        f"{bad_f1:.4f}"
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=class_names,
        zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(cm)

    if evaluation_dir is not None:
        save_evaluation_results(
            evaluation_dir=evaluation_dir,
            y_test=y_test,
            y_pred=y_pred,

            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            bad_recall=bad_recall,
            bad_f1=bad_f1,

            report_dict=report_dict,
            cm=cm,
            class_names=class_names,
            cv_results_df=cv_results_df,
            best_params=best_params,
            split_summary=split_summary
        )

        print(f"\nEvaluation results saved to: {evaluation_dir}")

    X_all = df[feature_columns]
    y_all = df[target_column].map(LABEL_TO_INT)

    final_imputer = SimpleImputer(strategy="median")
    X_all_imputed = final_imputer.fit_transform(X_all)

    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all_imputed)

    final_knn = KNeighborsClassifier(**best_params)
    final_knn.fit(X_all_scaled, y_all)

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    model_pack = {
        "imputer": final_imputer,
        "scaler": final_scaler,
        "model": final_knn,
        "feature_names": feature_names,
        "classes": ["Bad", "Good"],
        "model_numeric_classes": list(final_knn.classes_),
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "split_method": split_method,
        "evaluation_accuracy":
            float(accuracy),

        "evaluation_balanced_accuracy":
            float(balanced_accuracy),

        "evaluation_bad_recall":
            float(bad_recall),

        "evaluation_bad_f1":
            float(bad_f1),
        "best_params": best_params,
        "cv_results": cv_results_df.to_dict(orient="records")
    }

    joblib.dump(model_pack, model_output_path)

    print(f"\nPersonal model saved to: {model_output_path}")

    return model_pack