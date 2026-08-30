import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score
)

from training.train_personal_model import (
    split_personal_dataset_by_segment_train_test
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# CONFIGURATION
# ============================================================

LABEL_COLUMN = "Label"

METADATA_COLUMNS = {
    "Label",
    "Image_ID",
    "Segment_ID",
    "User_ID"
}

K_VALUES = [3, 5, 7, 9, 11, 15, 21, 31, 41, 51]

WEIGHTS_VALUES = [
    "uniform",
    "distance"
]

DISTANCE_METRICS = [
    "euclidean",
    "manhattan"
]

MAX_CV_SPLITS = 5

# Correlation above this value is considered very high.
CORRELATION_THRESHOLD = 0.95

STRUCTURAL_REDUNDANT_FEATURES = {
    "theta_RS_proj",
    "theta_LS_proj",
    "theta_RS_proj_relShoulders",
    "theta_LS_proj_relShoulders",
}


def build_initial_candidate_features(
    common_features
):
    """
    Start from features available for ALL personalized users,
    then remove only clear structural redundancies.
    """

    candidate_features = [
        feature
        for feature in common_features
        if feature
        not in STRUCTURAL_REDUNDANT_FEATURES
    ]

    print("\n" + "=" * 70)
    print("INITIAL FEATURE CANDIDATES")
    print("=" * 70)

    print(
        f"Common features: "
        f"{len(common_features)}"
    )

    print(
        f"Removed clear redundancies: "
        f"{len(STRUCTURAL_REDUNDANT_FEATURES)}"
    )

    print(
        f"Candidate features: "
        f"{len(candidate_features)}"
    )

    print(candidate_features)

    return candidate_features



def get_feature_columns(df):
    """
    Return all model-input columns and exclude metadata.
    """

    return [
        col
        for col in df.columns
        if col not in METADATA_COLUMNS
    ]

def find_highly_correlated_pairs(
    df,
    feature_columns,
    threshold=CORRELATION_THRESHOLD
):
    """
    Find feature pairs with absolute Pearson correlation
    greater than or equal to the chosen threshold.

    IMPORTANT:
    This function should be called only on training/CV data,
    never on the held-out test set.
    """

    X = df[feature_columns].copy()

    correlation_matrix = X.corr().abs()

    pairs = []

    for i in range(len(feature_columns)):
        for j in range(i + 1, len(feature_columns)):

            feature_a = feature_columns[i]
            feature_b = feature_columns[j]

            corr = correlation_matrix.loc[
                feature_a,
                feature_b
            ]

            if pd.isna(corr):
                continue

            if corr >= threshold:
                pairs.append({
                    "Feature_A": feature_a,
                    "Feature_B": feature_b,
                    "Abs_Correlation": float(corr)
                })

    pairs_df = pd.DataFrame(pairs)

    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(
            "Abs_Correlation",
            ascending=False
        ).reset_index(drop=True)

    return pairs_df

def evaluate_fold(
    train_df,
    val_df,
    feature_columns,
    params
):
    """
    Train and evaluate one KNN fold.

    Imputation and scaling are fitted ONLY on the training fold.
    """

    X_train = train_df[feature_columns]
    y_train = train_df[LABEL_COLUMN]

    X_val = val_df[feature_columns]
    y_val = val_df[LABEL_COLUMN]

    # -----------------------------------------
    # Missing-value imputation
    # -----------------------------------------

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    # -----------------------------------------
    # Scaling
    # -----------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # -----------------------------------------
    # KNN
    # -----------------------------------------

    model = KNeighborsClassifier(
        n_neighbors=params["n_neighbors"],
        weights=params["weights"],
        metric=params["metric"]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    return {
        "accuracy": accuracy_score(
            y_val,
            y_pred
        ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_val,
                y_pred
            ),

        "bad_recall":
            recall_score(
                y_val,
                y_pred,
                pos_label="Bad",
                zero_division=0
            ),

        "bad_f1":
            f1_score(
                y_val,
                y_pred,
                pos_label="Bad",
                zero_division=0
            )
    }

def evaluate_feature_subset_personal(
    train_cv_df,
    feature_columns,
    max_cv_splits=MAX_CV_SPLITS
):
    """
    Evaluate one feature subset on ONE personalized user.

    Only Train/CV segments are supplied to this function.
    Segment_ID is used as the grouping variable.
    """

    groups = train_cv_df["Segment_ID"]
    y = train_cv_df[LABEL_COLUMN]

    n_groups = groups.nunique()

    if n_groups < 2:
        raise RuntimeError(
            "Need at least two Segment_ID groups."
        )

    n_splits = min(
        max_cv_splits,
        n_groups
    )

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    configuration_results = []

    for k in K_VALUES:

        for weights in WEIGHTS_VALUES:

            for metric in DISTANCE_METRICS:

                fold_results = []

                params = {
                    "n_neighbors": k,
                    "weights": weights,
                    "metric": metric
                }

                for train_idx, val_idx in cv.split(
                    train_cv_df,
                    y,
                    groups=groups
                ):

                    fold_train = (
                        train_cv_df.iloc[train_idx]
                    )

                    fold_val = (
                        train_cv_df.iloc[val_idx]
                    )

                    if k > len(fold_train):
                        continue

                    result = evaluate_fold(
                        train_df=fold_train,
                        val_df=fold_val,
                        feature_columns=feature_columns,
                        params=params
                    )

                    fold_results.append(result)

                if len(fold_results) == 0:
                    continue

                configuration_results.append({

                    "K": k,

                    "Weights": weights,

                    "Metric": metric,

                    "Mean_Accuracy":
                        np.mean([
                            r["accuracy"]
                            for r in fold_results
                        ]),

                    "Std_Accuracy":
                        np.std([
                            r["accuracy"]
                            for r in fold_results
                        ]),

                    "Mean_Balanced_Accuracy":
                        np.mean([
                            r["balanced_accuracy"]
                            for r in fold_results
                        ]),

                    "Mean_Bad_Recall":
                        np.mean([
                            r["bad_recall"]
                            for r in fold_results
                        ]),

                    "Mean_Bad_F1":
                        np.mean([
                            r["bad_f1"]
                            for r in fold_results
                        ])
                })

    results_df = pd.DataFrame(
        configuration_results
    )

    results_df = results_df.sort_values(
        by=[
            "Mean_Balanced_Accuracy",
            "Mean_Bad_F1",
            "Std_Accuracy"
        ],
        ascending=[
            False,
            False,
            True
        ]
    ).reset_index(drop=True)

    best = results_df.iloc[0]

    return {
        "accuracy":
            float(best["Mean_Accuracy"]),

        "accuracy_std":
            float(best["Std_Accuracy"]),

        "balanced_accuracy":
            float(best["Mean_Balanced_Accuracy"]),

        "bad_recall":
            float(best["Mean_Bad_Recall"]),

        "bad_f1":
            float(best["Mean_Bad_F1"]),

        "k":
            int(best["K"]),

        "weights":
            best["Weights"],

        "metric":
            best["Metric"]
    }

def preload_personal_train_cv(personal_users):
    """
    Load and split every personalized user exactly once.
    Returns:
        dict: user_id -> train_cv_df
    """

    train_cv_data = {}

    for user_id, dataset_path in personal_users.items():

        print(f"Loading Train/CV data for {user_id}...")

        train_cv_df = load_personal_train_cv(
            dataset_path
        )

        train_cv_data[user_id] = train_cv_df

    return train_cv_data

def backward_feature_selection_all_users(
    train_cv_data,
    initial_features,
    min_features=20
):
    """
    Global backward feature selection for the
    personalized-model architecture.

    The same feature subset is evaluated across ALL users.

    Test segments are NEVER used.
    """

    current_features = list(
        initial_features
    )

    history = []

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------

    baseline = (
        evaluate_feature_subset_all_users(
            train_cv_data,
            current_features
        )
    )

    history.append({
        "Num_Features":
            len(current_features),

        "Removed_Feature":
            None,

        "Mean_Accuracy":
            baseline["mean_accuracy"],

        "Std_Between_Users":
            baseline["std_between_users"],

        "Mean_Balanced_Accuracy":
            baseline[
                "mean_balanced_accuracy"
            ],

        "Mean_Bad_Recall":
            baseline["mean_bad_recall"],

        "Mean_Bad_F1":
            baseline["mean_bad_f1"],

        "Features":
            "|".join(current_features)
    })

    # --------------------------------------------------
    # Save checkpoint after every completed iteration
    # --------------------------------------------------

    checkpoint_path = os.path.join(
        PROJECT_ROOT,
        "feature_selection_results",
        "personalized",
        "backward_selection_checkpoint.csv"
    )

    pd.DataFrame(history).to_csv(
        checkpoint_path,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        f"START: "
        f"{len(current_features)} features"
    )

    print(
        f"Mean CV accuracy: "
        f"{baseline['mean_accuracy']:.4f}"
    )

    print("=" * 70)

    # --------------------------------------------------
    # Backward elimination
    # --------------------------------------------------

    while len(current_features) > min_features:

        candidates = []

        print(
            f"\nTesting removals from "
            f"{len(current_features)} features..."
        )

        for feature_to_remove in (
            current_features
        ):

            candidate_features = [
                f
                for f in current_features
                if f != feature_to_remove
            ]

            result = (
                evaluate_feature_subset_all_users(
                    train_cv_data,
                    candidate_features
                )
            )

            candidates.append({
                "removed":
                    feature_to_remove,

                "features":
                    candidate_features,

                **result
            })

            print(
                f"Remove "
                f"{feature_to_remove:35s} "
                f"→ "
                f"{result['mean_accuracy']:.4f}"
            )

        # --------------------------------------------------
        # Select best removal
        # --------------------------------------------------

        candidates.sort(
            key=lambda x: (
                -x["mean_balanced_accuracy"],
                -x["mean_bad_f1"],
                x["std_between_users"]
            )
        )

        best = candidates[0]

        current_features = (
            best["features"]
        )

        history.append({

            "Num_Features":
                len(current_features),

            "Removed_Feature":
                best["removed"],

            "Mean_Accuracy":
                best["mean_accuracy"],

            "Std_Between_Users":
                best["std_between_users"],

            "Mean_Balanced_Accuracy":
                best[
                    "mean_balanced_accuracy"
                ],

            "Mean_Bad_Recall":
                best["mean_bad_recall"],

            "Mean_Bad_F1":
                best["mean_bad_f1"],

            "Features":
                "|".join(
                    current_features
                )
        })

        # Checkpoint
        pd.DataFrame(history).to_csv(
            checkpoint_path,
            index=False
        )

        print("\nBEST REMOVAL:")
        print(best["removed"])

        print(
            f"Remaining features: "
            f"{len(current_features)}"
        )

        print(
            f"Mean CV accuracy: "
            f"{best['mean_accuracy']:.4f}"
        )

        print(
            f"Mean Bad Recall: "
            f"{best['mean_bad_recall']:.4f}"
        )

    return pd.DataFrame(history)

def analyze_personal_user(
    user_id,
    dataset_path,
    output_root
):
    print("\n" + "=" * 70)
    print(f"USER: {user_id}")
    print("=" * 70)

    print(f"Dataset path: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found for user '{user_id}':\n"
            f"{dataset_path}"
        )

    df = pd.read_excel(dataset_path)

    df.columns = (
        df.columns
        .str.strip()
    )

    df = df[
        df["Label"].isin(
            ["Good", "Bad"]
        )
    ].copy()

    # EXACT same train/test splitting logic
    # as the final personalized model.
    train_cv_df, _ = (
        split_personal_dataset_by_segment_train_test(
            df,
            test_segment_ratio=0.25,
            random_state=42
        )
    )

    feature_columns = (
        get_feature_columns(df)
    )

    print(
        f"\nTotal candidate features: "
        f"{len(feature_columns)}"
    )

    # ----------------------------------------
    # Correlation analysis
    # TRAIN/CV ONLY
    # ----------------------------------------

    correlated_pairs = (
        find_highly_correlated_pairs(
            train_cv_df,
            feature_columns
        )
    )

    user_output_dir = os.path.join(
        output_root,
        user_id
    )

    os.makedirs(
        user_output_dir,
        exist_ok=True
    )

    correlation_path = os.path.join(
        user_output_dir,
        "high_correlation_pairs.csv"
    )

    correlated_pairs.to_csv(
        correlation_path,
        index=False
    )

    print(
        "\nHighly correlated pairs:"
    )

    if correlated_pairs.empty:

        print(
            "No pairs above threshold."
        )

    else:

        print(
            correlated_pairs.to_string(
                index=False
            )
        )

    print(
        f"\nSaved to: "
        f"{correlation_path}"
    )

    return correlated_pairs

def load_personal_train_cv(
    dataset_path
):
    """
    Load one personalized dataset and return ONLY
    its Train/CV pool.

    Held-out test segments are never used during
    feature selection.
    """

    df = pd.read_excel(dataset_path)

    df.columns = df.columns.str.strip()

    df = df[
        df["Label"].isin(
            ["Good", "Bad"]
        )
    ].copy()

    train_cv_df, _ = (
        split_personal_dataset_by_segment_train_test(
            df,
            test_segment_ratio=0.25,
            random_state=42
        )
    )

    return train_cv_df

def evaluate_feature_subset_all_users(
    train_cv_data,
    feature_columns
):
    """
    Evaluate the same feature subset for all users.

    Every user receives equal weight.
    Train/CV datasets are already preloaded.
    """

    user_results = []

    for user_id, train_cv_df in train_cv_data.items():

        result = evaluate_feature_subset_personal(
            train_cv_df,
            feature_columns
        )

        user_results.append({
            "User_ID": user_id,
            **result
        })

    results_df = pd.DataFrame(
        user_results
    )

    return {
        "mean_accuracy":
            results_df["accuracy"].mean(),

        "std_between_users":
            results_df["accuracy"].std(),

        "mean_balanced_accuracy":
            results_df[
                "balanced_accuracy"
            ].mean(),

        "mean_bad_recall":
            results_df[
                "bad_recall"
            ].mean(),

        "mean_bad_f1":
            results_df[
                "bad_f1"
            ].mean(),

        "per_user":
            results_df
    }

def compare_user_feature_schemas(personal_users):
    """
    Compare the feature columns available in all personalized datasets.

    Returns:
        sorted list of features that exist for ALL users.
    """

    feature_sets = {}

    for user_id, dataset_path in personal_users.items():

        df = pd.read_excel(dataset_path)
        df.columns = df.columns.str.strip()

        features = set(
            get_feature_columns(df)
        )

        feature_sets[user_id] = features

        print(
            f"{user_id}: "
            f"{len(features)} features"
        )

    # Features available for every user
    common_features = set.intersection(
        *feature_sets.values()
    )

    # Union of every feature appearing anywhere
    all_features = set.union(
        *feature_sets.values()
    )

    print("\n" + "=" * 70)
    print("COMMON FEATURES")
    print("=" * 70)

    print(
        f"Number of common features: "
        f"{len(common_features)}"
    )

    print(sorted(common_features))

    print("\n" + "=" * 70)
    print("FEATURE DIFFERENCES")
    print("=" * 70)

    for user_id, features in feature_sets.items():

        missing = sorted(
            all_features - features
        )

        extra_vs_common = sorted(
            features - common_features
        )

        print(f"\n{user_id}")

        print(
            "Missing compared with union:",
            missing
        )

        print(
            "Extra compared with common set:",
            extra_vs_common
        )

    return sorted(common_features)

def select_final_feature_subset(selection_history):

    ranked = selection_history.sort_values(
        by=[
            "Mean_Balanced_Accuracy",
            "Mean_Bad_F1",
            "Std_Between_Users",
            "Num_Features"
        ],
        ascending=[
            False,
            False,
            True,
            True
        ]
    ).reset_index(drop=True)

    best = ranked.iloc[0]

    selected_features = [
        feature
        for feature in best["Features"].split("|")
        if feature
    ]

    return best, selected_features


def save_feature_selection_results(
    selection_history,
    output_root
):
    """
    Save report-ready results of the personalized feature-selection process.

    Outputs:
        1. performance_vs_num_features.png
        2. bad_class_metrics_vs_num_features.png
        3. selected_features.csv
        4. feature_selection_summary.csv
        5. feature_selection_report_text.txt
    """

    os.makedirs(
        output_root,
        exist_ok=True
    )

    history = (
        selection_history
        .sort_values(
            "Num_Features",
            ascending=False
        )
        .reset_index(drop=True)
        .copy()
    )

    # ============================================================
    # Select final feature subset
    # ============================================================

    best, selected_features = (
        select_final_feature_subset(history)
    )

    selected_num_features = int(
        best["Num_Features"]
    )

    selected_accuracy = float(
        best["Mean_Accuracy"]
    )

    selected_std = float(
        best["Std_Between_Users"]
    )

    selected_balanced_accuracy = float(
        best["Mean_Balanced_Accuracy"]
    )

    selected_bad_recall = float(
        best["Mean_Bad_Recall"]
    )

    selected_bad_f1 = float(
        best["Mean_Bad_F1"]
    )

    initial_num_features = int(
        history.iloc[0]["Num_Features"]
    )

    # ============================================================
    # FIGURE 1:
    # Overall classification performance
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history["Num_Features"],
        history["Mean_Accuracy"],
        marker="o",
        label="Mean Accuracy"
    )

    ax.plot(
        history["Num_Features"],
        history["Mean_Balanced_Accuracy"],
        marker="o",
        label="Mean Balanced Accuracy"
    )

    ax.scatter(
        selected_num_features,
        selected_balanced_accuracy,
        s=90,
        zorder=5,
        label="Selected subset"
    )

    ax.set_xlabel(
        "Number of Features"
    )

    ax.set_ylabel(
        "Cross-Validation Score"
    )

    ax.set_title(
        "Personalized Model Feature Selection"
    )

    ax.set_ylim(
        0.0,
        1.0
    )

    # Show the elimination process from many → few features
    ax.invert_xaxis()

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    performance_plot_path = os.path.join(
        output_root,
        "performance_vs_num_features.png"
    )

    fig.savefig(
        performance_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ============================================================
    # FIGURE 2:
    # Performance specifically on the Bad class
    # ============================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history["Num_Features"],
        history["Mean_Bad_Recall"],
        marker="o",
        label="Bad Recall"
    )

    ax.plot(
        history["Num_Features"],
        history["Mean_Bad_F1"],
        marker="o",
        label="Bad F1"
    )

    ax.set_xlabel(
        "Number of Features"
    )

    ax.set_ylabel(
        "Cross-Validation Score"
    )

    ax.set_title(
        "Bad-Posture Detection During Feature Selection"
    )

    ax.set_ylim(
        0.0,
        1.0
    )

    ax.invert_xaxis()

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    bad_metrics_plot_path = os.path.join(
        output_root,
        "bad_class_metrics_vs_num_features.png"
    )

    fig.savefig(
        bad_metrics_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ============================================================
    # Save selected feature list
    # ============================================================

    selected_features_df = pd.DataFrame({
        "Feature": selected_features
    })

    selected_features_path = os.path.join(
        output_root,
        "selected_features.csv"
    )

    selected_features_df.to_csv(
        selected_features_path,
        index=False
    )

    # ============================================================
    # Save concise numerical summary
    # ============================================================

    summary_df = pd.DataFrame([{
        "Initial_Num_Features":
            initial_num_features,

        "Selected_Num_Features":
            selected_num_features,

        "Mean_CV_Accuracy":
            selected_accuracy,

        "Std_Between_Users":
            selected_std,

        "Mean_Balanced_Accuracy":
            selected_balanced_accuracy,

        "Mean_Bad_Recall":
            selected_bad_recall,

        "Mean_Bad_F1":
            selected_bad_f1
    }])

    summary_path = os.path.join(
        output_root,
        "feature_selection_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ============================================================
    # Generate report-ready English text
    # ============================================================

    report_text = f"""
Feature Selection

To reduce redundant and potentially uninformative dimensions in the
personalized KNN classifier, feature selection was performed using
correlation-aware backward sequential feature elimination.

The procedure started with {initial_num_features} candidate features.
Clear structural redundancies were removed before the wrapper-based
selection stage. The remaining features were evaluated using grouped
cross-validation, where Segment_ID was used as the grouping variable.
This ensured that frames originating from the same recorded posture
segment were never divided between the training and validation portions
of the same fold.

At every backward-selection iteration, each remaining feature was
removed individually. For every candidate subset, KNN hyperparameters
were re-optimized using the predefined search space. Median imputation
and feature standardization were fitted exclusively on the training
portion of each cross-validation fold.

The same candidate feature subset was evaluated independently for all
personalized users, and the final score was computed as the unweighted
mean of the per-user cross-validation scores. Consequently, every user
contributed equally to the feature-selection decision regardless of
the number of recorded frames.

The held-out test segments were excluded from the entire feature-selection
procedure.

Along the backward-elimination path, the selected representation contained
{selected_num_features} features. It achieved a mean cross-validation
accuracy of {selected_accuracy:.3f}, a mean balanced accuracy of
{selected_balanced_accuracy:.3f}, a mean Bad-class recall of
{selected_bad_recall:.3f}, and a mean Bad-class F1 score of
{selected_bad_f1:.3f}. The standard deviation of mean accuracy across
users was {selected_std:.3f}.

The final subset was selected according to the highest mean balanced
cross-validation accuracy across users. Bad-class F1 score was used as
a secondary criterion, followed by lower between-user variability and,
in case of remaining ties, the smaller feature representation.
""".strip()

    report_text_path = os.path.join(
        output_root,
        "feature_selection_report_text.txt"
    )

    with open(
        report_text_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report_text)

    # ============================================================
    # Console summary
    # ============================================================

    print("\n" + "=" * 70)
    print("FINAL FEATURE-SELECTION RESULT")
    print("=" * 70)

    print(
        f"Initial features: "
        f"{initial_num_features}"
    )

    print(
        f"Selected features: "
        f"{selected_num_features}"
    )

    print(
        f"Mean CV Accuracy: "
        f"{selected_accuracy:.4f}"
    )

    print(
        f"Mean Balanced Accuracy: "
        f"{selected_balanced_accuracy:.4f}"
    )

    print(
        f"Mean Bad Recall: "
        f"{selected_bad_recall:.4f}"
    )

    print(
        f"Mean Bad F1: "
        f"{selected_bad_f1:.4f}"
    )

    print(
        f"Std Between Users: "
        f"{selected_std:.4f}"
    )

    print("\nSelected features:")

    for feature in selected_features:
        print(
            f"  - {feature}"
        )

    print("\nReport outputs:")
    print(performance_plot_path)
    print(bad_metrics_plot_path)
    print(selected_features_path)
    print(summary_path)
    print(report_text_path)

    return {
        "best_row":
            best,

        "selected_features":
            selected_features,

        "performance_plot_path":
            performance_plot_path,

        "bad_metrics_plot_path":
            bad_metrics_plot_path,

        "summary_path":
            summary_path,

        "report_text_path":
            report_text_path
    }

if __name__ == "__main__":

    PERSONAL_USERS = {
        "shirel": os.path.join(
            PROJECT_ROOT,
            "users",
            "shirel",
            "calibration_dataset.xlsx"
        ),

        "shirel_2": os.path.join(
            PROJECT_ROOT,
            "users",
            "shirel",
            "calibration_dataset.xlsx"
        ),

        "avigail": os.path.join(
            PROJECT_ROOT,
            "users",
            "avigail",
            "calibration_dataset.xlsx"
        ),

        "naomi": os.path.join(
            PROJECT_ROOT,
            "users",
            "naomi",
            "calibration_dataset.xlsx"
        ),

        "ester_fradkin": os.path.join(
            PROJECT_ROOT,
            "users",
            "ester_fradkin",
            "calibration_dataset.xlsx"
        ),

        "oria": os.path.join(
            PROJECT_ROOT,
            "users",
            "oria",
            "calibration_dataset.xlsx"
        ),

        "sophie_fradkin": os.path.join(
            PROJECT_ROOT,
            "users",
            "sophie_fradkin",
            "calibration_dataset.xlsx"
        ),

        "tegenu": os.path.join(
            PROJECT_ROOT,
            "users",
            "tegenu",
            "calibration_dataset.xlsx"
        ),

        "yarin_levi": os.path.join(
            PROJECT_ROOT,
            "users",
            "yarin_levi",
            "calibration_dataset.xlsx"
        ),
    }

    OUTPUT_ROOT = os.path.join(
        PROJECT_ROOT,
        "feature_selection_results",
        "personalized"
    )

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    # ============================================================
    # 1. Verify common feature representation
    # ============================================================

    common_features = compare_user_feature_schemas(
        PERSONAL_USERS
    )

    # ============================================================
    # 2. Remove only clear structural redundancies
    # ============================================================

    candidate_features = (
        build_initial_candidate_features(
            common_features
        )
    )

    # ============================================================
    # 3. Global backward feature selection
    # ============================================================

    print("\n" + "=" * 70)
    print("STARTING GLOBAL BACKWARD FEATURE SELECTION")
    print("=" * 70)

    train_cv_data = preload_personal_train_cv(
        PERSONAL_USERS
    )

    selection_history = backward_feature_selection_all_users(
        train_cv_data=train_cv_data,
        initial_features=candidate_features,
        min_features=20
    )

    # ============================================================
    # 4. Save complete selection history
    # ============================================================

    selection_output_path = os.path.join(
        OUTPUT_ROOT,
        "backward_selection_history.csv"
    )

    selection_history.to_csv(
        selection_output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FEATURE SELECTION FINISHED")
    print("=" * 70)

    print(
        f"Results saved to:\n"
        f"{selection_output_path}"
    )

    print("\nSelection history:")

    print(
        selection_history[
            [
                "Num_Features",
                "Removed_Feature",
                "Mean_Accuracy",
                "Std_Between_Users",
                "Mean_Balanced_Accuracy",
                "Mean_Bad_Recall",
                "Mean_Bad_F1"
            ]
        ].to_string(index=False)
    )

    report_results = save_feature_selection_results(
        selection_history=selection_history,
        output_root=OUTPUT_ROOT
    )

