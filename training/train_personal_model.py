# ============================================================
# TRAIN PERSONAL KNN MODEL — GROUP CROSS-VALIDATION VERSION
# ============================================================

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shutil

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from training.permutation_importance import run_permutation_importance

LABEL_TO_INT = {"Bad": 0, "Good": 1}
INT_TO_LABEL = {0: "Bad", 1: "Good"}

def split_personal_dataset_by_segment_train_test(
    df,
    test_segment_ratio=0.25,
    random_state=42
):
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
    if k_values is None:
        k_values = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51]

    unique_groups = pd.Series(groups).dropna().unique()
    num_groups = len(unique_groups)

    if num_groups < 2:
        raise RuntimeError(
            "GroupKFold requires at least 2 unique Segment_ID groups."
        )

    n_splits = min(max_cv_splits, num_groups)
    group_kfold = GroupKFold(n_splits=n_splits)

    results = []

    for k in k_values:
        for weights in ["uniform", "distance"]:
            for metric in ["euclidean", "manhattan"]:

                fold_accuracies = []
                skipped_folds = 0

                for train_idx, val_idx in group_kfold.split(X, y, groups=groups):
                    X_fold_train = X.iloc[train_idx]
                    y_fold_train = y.iloc[train_idx]

                    X_fold_val = X.iloc[val_idx]
                    y_fold_val = y.iloc[val_idx]

                    if k > len(y_fold_train):
                        skipped_folds += 1
                        continue

                    imputer = SimpleImputer(strategy="median")
                    X_fold_train_imputed = imputer.fit_transform(X_fold_train)
                    X_fold_val_imputed = imputer.transform(X_fold_val)

                    scaler = StandardScaler()
                    X_fold_train_scaled = scaler.fit_transform(X_fold_train_imputed)
                    X_fold_val_scaled = scaler.transform(X_fold_val_imputed)

                    candidate_model = KNeighborsClassifier(
                        n_neighbors=k,
                        weights=weights,
                        metric=metric
                    )

                    candidate_model.fit(X_fold_train_scaled, y_fold_train)
                    fold_pred = candidate_model.predict(X_fold_val_scaled)

                    fold_accuracy = accuracy_score(y_fold_val, fold_pred)
                    fold_accuracies.append(fold_accuracy)

                if not fold_accuracies:
                    continue

                results.append({
                    "n_neighbors": k,
                    "weights": weights,
                    "metric": metric,
                    "mean_cv_accuracy": float(np.mean(fold_accuracies)),
                    "std_cv_accuracy": float(np.std(fold_accuracies)),
                    "num_folds_used": len(fold_accuracies),
                    "num_folds_skipped": skipped_folds
                })

    if not results:
        raise RuntimeError(
            "No valid KNN hyperparameter configurations were evaluated."
        )

    cv_results_df = pd.DataFrame(results)
    cv_results_df = cv_results_df.sort_values(
        by=["mean_cv_accuracy", "std_cv_accuracy"],
        ascending=[False, True]
    ).reset_index(drop=True)

    best_row = cv_results_df.iloc[0]

    best_params = {
        "n_neighbors": int(best_row["n_neighbors"]),
        "weights": best_row["weights"],
        "metric": best_row["metric"]
    }

    return best_params, cv_results_df


def save_k_cv_search_figure(cv_results_df, evaluation_dir):
    hyper_dir = os.path.join(evaluation_dir, "hyperparameter_selection")
    os.makedirs(hyper_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    grouped = cv_results_df.groupby(["metric", "weights"])

    for (metric, weights), group_df in grouped:
        group_df = group_df.sort_values("n_neighbors")
        label = f"{metric} + {weights}"

        ax.plot(
            group_df["n_neighbors"],
            group_df["mean_cv_accuracy"],
            marker="o",
            label=label
        )

    ax.set_title("K vs Mean Group-CV Accuracy")
    ax.set_xlabel("Number of Neighbors (K)")
    ax.set_ylabel("Mean Group-CV Accuracy")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    figure_path = os.path.join(hyper_dir, "K_vs_group_cv_accuracy.png")
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
    report_dict,
    cm,
    class_names,
    cv_results_df=None,
    best_params=None,
    split_summary=None
):
    os.makedirs(evaluation_dir, exist_ok=True)

    report_text = classification_report(y_test, y_pred, zero_division=0)

    with open(os.path.join(evaluation_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")

        if best_params is not None:
            f.write(f"Best parameters chosen by GroupKFold CV: {best_params}\n\n")

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
    if split_method != "segment":
        raise ValueError(
            "This GroupKFold version expects split_method='segment'."
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
        f"Split method: segment GroupKFold CV\n"
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
    metadata_columns = [target_column, "Image_ID", "Segment_ID"]

# first - all 50 features
    feature_columns = [
        col for col in df.columns
        if col not in metadata_columns
    ]

    # 2 - only egineered features
    # feature_columns = [
    #     "headForwardDepth",
    #     "headHeight",
    #     "headTurn",
    #     "eyeWidth",
    #
    #     "shoulderTilt",
    #     "torsoRotation",
    #     "shoulderWidth",
    #
    #     "theta_shoulders",
    #     "thetaNeck",
    #     "thetaNeck_rel",
    #
    #     "theta_RS_proj",
    #     "theta_LS_proj",
    #     "theta_RS_proj_relShoulders",
    #     "theta_LS_proj_relShoulders",
    #
    #     "theta_ears",
    #     "theta_ears_rel",
    #
    #     "theta_LEar_LS",
    #     "theta_LEar_LS_relShoulders",
    #     "theta_REar_RS",
    #     "theta_REar_RS_relShoulders",
    #
    #     "leftEarShoulderDistance",
    #     "rightEarShoulderDistance",
    # ]

    # 3- Engineered + Z
    # feature_columns = [
    #     "headForwardDepth",
    #     "headHeight",
    #     "headTurn",
    #     "eyeWidth",
    #
    #     "shoulderTilt",
    #     "torsoRotation",
    #     "shoulderWidth",
    #
    #     "theta_shoulders",
    #     "thetaNeck",
    #     "thetaNeck_rel",
    #
    #     "theta_RS_proj",
    #     "theta_LS_proj",
    #     "theta_RS_proj_relShoulders",
    #     "theta_LS_proj_relShoulders",
    #
    #     "theta_ears",
    #     "theta_ears_rel",
    #
    #     "theta_LEar_LS",
    #     "theta_LEar_LS_relShoulders",
    #     "theta_REar_RS",
    #     "theta_REar_RS_relShoulders",
    #
    #     "leftEarShoulderDistance",
    #     "rightEarShoulderDistance",
    #
    #     "nose_z",
    #     "leftEar_z",
    #     "rightEar_z",
    #     "leftShoulder_z",
    #     "rightShoulder_z",
    #     "leftEye_z",
    #     "rightEye_z",
    # ]

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

    print("\nBest parameters chosen by GroupKFold CV:")
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

    accuracy = accuracy_score(y_test, y_pred)
    class_names = ["Bad", "Good"]

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

    print(f"\nTest Accuracy: {accuracy:.2%}")

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
        run_permutation_importance(
            knn_model=evaluation_knn,
            imputer=evaluation_imputer,
            scaler=evaluation_scaler,
            X_test=X_test,
            y_test=y_test.map(LABEL_TO_INT),
            output_dir=os.path.join(
                evaluation_dir,
                "feature_importance"
            ),
            n_repeats=50
        )
        save_evaluation_results(
            evaluation_dir=evaluation_dir,
            y_test=y_test,
            y_pred=y_pred,
            accuracy=accuracy,
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
        "evaluation_accuracy": accuracy,
        "best_params": best_params,
        "cv_results": cv_results_df.to_dict(orient="records")
    }

    joblib.dump(model_pack, model_output_path)

    print(f"\nPersonal model saved to: {model_output_path}")

    return model_pack