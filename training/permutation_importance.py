import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


def run_permutation_importance(
    knn_model,
    imputer,
    scaler,
    X_test,
    y_test,
    output_dir,
    n_repeats=20,
    random_state=42,
):
    """
    Computes permutation importance on the unseen test set.

    Important:
    - Uses the already trained imputer, scaler, and KNN model.
    - Does not refit anything.
    - Measures how much accuracy drops when each feature is shuffled.
    """

    os.makedirs(output_dir, exist_ok=True)

    pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("knn", knn_model),
    ])

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="accuracy",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": result.importances_mean,
        "Importance_STD": result.importances_std,
    })

    importance_df = importance_df.sort_values(
        by="Importance_Mean",
        ascending=False
    )

    csv_path = os.path.join(output_dir, "permutation_importance.csv")
    importance_df.to_csv(csv_path, index=False)

    # Plot top 20 features
    top_features = importance_df.head(20)

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance_Mean"][::-1],
        xerr=top_features["Importance_STD"][::-1],
    )
    plt.xlabel("Accuracy decrease after shuffling")
    plt.title("Permutation Importance - Top Features")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "permutation_importance_top20.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("\nPermutation Importance saved:")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")

    print("\nTop 15 important features:")
    print(importance_df.head(15))

    print("\nFeatures with zero or negative importance:")
    print(importance_df[importance_df["Importance_Mean"] <= 0])

    return importance_df