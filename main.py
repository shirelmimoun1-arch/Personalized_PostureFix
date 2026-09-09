# ============================================================
# MAIN – PERSONALIZED AI POSTURE MONITOR
# End-to-end orchestration of calibration, training, and live inference
# ============================================================
#
# Purpose
# -------
# Provide the main entry point for the personalized posture-monitoring
# system and coordinate the complete user-specific workflow.
#
# The module connects the calibration, dataset construction, personalized
# model training, and live monitoring components into a single interactive
# command-line pipeline.
#
#
# System workflow
# ---------------
# For each user, the program can execute one of several supported flows:
#
# 1. Full calibration and training
#       - record a new personalized calibration session,
#       - create the calibration feature dataset,
#       - train and evaluate a personalized KNN model,
#       - start live posture monitoring.
#
# 2. Dataset reconstruction and retraining
#       - reuse previously recorded calibration frames and labels,
#       - regenerate the feature dataset,
#       - retrain the personalized model,
#       - start live monitoring.
#
# 3. Model retraining from an existing dataset
#       - skip calibration and dataset construction,
#       - retrain the personalized KNN model,
#       - start live monitoring.
#
# 4. Live monitoring only
#       - load an existing personalized model,
#       - start the real-time posture-monitoring interface.
#
#
# User-specific artifacts
# -----------------------
# Each user is associated with a dedicated directory containing:
#   - calibration frames,
#   - frame-level calibration labels,
#   - the derived calibration feature dataset,
#   - evaluation outputs.
#
# The trained personalized KNN model is stored separately under the
# user-model directory.
#
#
# Architectural role
# ------------------
# This module does not implement feature extraction, model training, or
# inference logic directly. Instead, it acts as an orchestration layer
# that invokes the specialized modules responsible for:
#
#   - calibration.frame_extraction
#   - calibration.create_user_dataset
#   - training.train_personal_model
#   - live.live_monitoring
#
# This separation keeps the end-to-end application flow independent from
# the internal implementation of each processing stage.
#
# ============================================================

import os

from calibration.frame_extraction import run_calibration
from calibration.create_user_dataset import create_user_dataset
from training.train_personal_model import train_personal_model
from live.live_monitoring import run_live_monitoring


# ============================================================
# CONFIGURATION
# ============================================================

BASE_USER_DIR = "users"
MODEL_DIR = "models/user_models"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_user_id():
    """
    Request and normalize a user identifier from interactive input.

    The returned identifier is converted to lowercase and whitespace is
    replaced by underscores so it can be used consistently in directory
    and model-file names.

    Returns
    -------
    str
        Normalized non-empty user identifier.
    """
    while True:
        user_id = input("Enter user name / ID: ").strip()

        if user_id:
            break

        print("Enter user name / ID:")

    return user_id.strip().replace(" ", "_").lower()


def get_user_paths(user_id):
    """
    Construct and initialize all filesystem paths associated with a user.

    The function defines locations for calibration frames, calibration
    labels, the derived feature dataset, evaluation outputs, and the
    serialized personalized model. Required parent directories are created
    if they do not already exist.

    Parameters
    ----------
    user_id : str
        Normalized identifier of the active user.

    Returns
    -------
    dict
        Mapping containing the user-specific directory and file paths used
        throughout the personalized pipeline.
    """
    user_dir = os.path.join(BASE_USER_DIR, user_id)

    paths = {
        "user_dir": user_dir,
        "frames_dir": os.path.join(user_dir, "frames"),
        "labels_csv": os.path.join(user_dir, "labels.csv"),
        "dataset_path": os.path.join(user_dir, "calibration_dataset.xlsx"),
        "model_path": os.path.join(MODEL_DIR, f"{user_id}_personal_knn.joblib"),
        "evaluation_dir": os.path.join(user_dir, "evaluation"),
    }

    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(paths["frames_dir"], exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    return paths


def model_exists(model_path):
    """
    Check whether a serialized personalized model exists.

    Parameters
    ----------
    model_path : str or Path
        Expected model-file path.

    Returns
    -------
    bool
        True if the model file exists, otherwise False.
    """
    return os.path.exists(model_path)


def dataset_exists(dataset_path):
    """
    Check whether a personalized calibration dataset exists.

    Parameters
    ----------
    dataset_path : str or Path
        Expected calibration-dataset path.

    Returns
    -------
    bool
        True if the dataset exists, otherwise False.
    """
    return os.path.exists(dataset_path)


def calibration_files_exist(paths):
    """
    Verify that reusable calibration inputs are available.

    A calibration session is considered reusable only when both the labels
    file and the frames directory exist and the frames directory contains
    at least one recorded frame.

    Parameters
    ----------
    paths : dict
        User-specific path mapping returned by ``get_user_paths``.

    Returns
    -------
    bool
        True when existing calibration frames and labels are available.
    """
    return (
        os.path.exists(paths["labels_csv"])
        and os.path.exists(paths["frames_dir"])
        and len(os.listdir(paths["frames_dir"])) > 0
    )


def ask_user_action(has_model, has_dataset, has_calibration_files):
    """
    Present the interactive pipeline menu and return a valid user action.

    Availability of each operation depends on the artifacts already
    present for the current user. Actions requiring missing inputs are
    displayed as unavailable and cannot be selected successfully.

    Parameters
    ----------
    has_model : bool
        Whether an existing personalized model is available.
    has_dataset : bool
        Whether an existing calibration feature dataset is available.
    has_calibration_files : bool
        Whether reusable calibration frames and labels are available.

    Returns
    -------
    str
        Internal action identifier corresponding to the selected workflow.
    """
    while True:
        print("\nChoose action:")
        print("1. Run new calibration, recreate dataset, and train a new model")

        if has_calibration_files:
            print("2. Recreate dataset from existing frames/labels and retrain model")
        else:
            print("2. Recreate dataset from existing frames/labels and retrain model (not available)")

        if has_dataset:
            print("3. Train/retrain model from existing dataset only")
        else:
            print("3. Train/retrain model from existing dataset only (not available)")

        if has_model:
            print("4. Start live monitoring with existing model")
        else:
            print("4. Start live monitoring with existing model (not available)")

        print("5. Enter a different user name / ID")
        print("6. Exit")

        choice = input("Enter choice (1/2/3/4/5/6): ").strip()

        if choice == "1":
            return "new_calibration"

        if choice == "2":
            if has_calibration_files:
                return "recreate_dataset_and_train"
            print("\nNo existing frames/labels found.")

        elif choice == "3":
            if has_dataset:
                return "train_existing_dataset"
            print("\nNo existing calibration dataset found.")

        elif choice == "4":
            if has_model:
                return "live_monitoring"
            print("\nNo existing personal model found.")

        elif choice == "5":
            return "change_user"

        elif choice == "6":
            return "exit"

        else:
            print("Invalid choice.")


def create_dataset_from_existing_calibration(paths):
    """
    Reconstruct the personalized feature dataset from existing calibration
    frames and labels.

    The function delegates dataset creation to ``create_user_dataset`` and
    writes the resulting feature table to the user-specific dataset path.

    Parameters
    ----------
    paths : dict
        User-specific path mapping returned by ``get_user_paths``.
    """
    print("\nCreating calibration dataset from existing frames and labels.")
    print("Frames folder:")
    print(paths["frames_dir"])
    print("Labels CSV:")
    print(paths["labels_csv"])

    create_user_dataset(
        frames_dir=paths["frames_dir"],
        labels_csv_path=paths["labels_csv"],
        output_file=paths["dataset_path"]
    )

    print("\nCalibration dataset created/updated.")
    print("Dataset path:")
    print(paths["dataset_path"])


def train_model_from_existing_dataset(paths):
    """
    Train and evaluate a personalized KNN model from an existing dataset.

    The function verifies that the calibration dataset is available and
    delegates the complete personalized training and evaluation procedure
    to ``train_personal_model``.

    Parameters
    ----------
    paths : dict
        User-specific path mapping returned by ``get_user_paths``.

    Returns
    -------
    bool
        True when model training completes successfully, otherwise False.
    """
    if not dataset_exists(paths["dataset_path"]):
        print("\nNo existing calibration dataset found.")
        print("Please create the dataset first.")
        return False

    print("\nTraining personal KNN model from calibration dataset.")
    print("Dataset path:")
    print(paths["dataset_path"])

    train_personal_model(
        dataset_path=paths["dataset_path"],
        model_output_path=paths["model_path"],
        frames_dir=paths["frames_dir"],
        evaluation_dir=paths["evaluation_dir"]
    )

    print("\nPersonalized KNN model created/updated.")
    print("Model path:")
    print(paths["model_path"])

    return True


def run_full_calibration_and_training(paths):
    """
    Execute the complete personalized calibration and training workflow.

    The procedure records a new calibration session, constructs the
    calibration feature dataset, and trains the personalized model.

    Parameters
    ----------
    paths : dict
        User-specific path mapping returned by ``get_user_paths``.

    Returns
    -------
    bool
        True when calibration-data processing and model training complete
        successfully, otherwise False.
    """
    print("\nStarting automatic calibration.")

    run_calibration(
        frames_dir=paths["frames_dir"],
        labels_csv_path=paths["labels_csv"],
        target_fps=5
    )

    create_dataset_from_existing_calibration(paths)
    return train_model_from_existing_dataset(paths)


def recreate_dataset_and_train(paths):
    """
    Rebuild the dataset from existing calibration data and retrain the model.

    This workflow is intended for cases where calibration frames and labels
    are already available but the derived dataset or trained model should
    be regenerated.

    Parameters
    ----------
    paths : dict
        User-specific path mapping returned by ``get_user_paths``.

    Returns
    -------
    bool
        True when dataset reconstruction and model retraining complete
        successfully, otherwise False.
    """
    if not calibration_files_exist(paths):
        print("\nNo existing frames/labels found.")
        print("Please run new calibration first.")
        return False

    create_dataset_from_existing_calibration(paths)
    return train_model_from_existing_dataset(paths)


def start_live_monitoring(paths):
    """
    Start real-time personalized posture monitoring for the active user.

    The function verifies that a trained model is available and then
    delegates inference and feedback generation to ``run_live_monitoring``.

    Parameters
    ----------
    paths : dict
        User-specific path mapping returned by ``get_user_paths``.
    """
    if not model_exists(paths["model_path"]):
        print("\nNo personal model found.")
        print("Please train a model first.")
        return

    print("\nStarting live personalized posture monitoring.")

    run_live_monitoring(
        model_path=paths["model_path"],
        calibration_dataset_path=paths["dataset_path"]
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """
    Run the interactive end-to-end personalized posture-monitoring pipeline.

    The function initializes the active user context, inspects the available
    user-specific artifacts, presents the supported workflow options, and
    dispatches execution to the corresponding calibration, training, or
    live-monitoring stage.

    Depending on the selected action, the pipeline may perform:
        - complete calibration and training,
        - dataset reconstruction and retraining,
        - retraining from an existing dataset,
        - live monitoring using an existing model.

    After successful training, live monitoring is started automatically.
    """
    while True:
        user_id = get_user_id()
        paths = get_user_paths(user_id)

        print("\n====================================")
        print("Personalized AI Posture Monitor")
        print("====================================")

        print(f"\nCurrent user: {user_id}")
        print(f"User folder: {paths['user_dir']}")
        print(f"Frames folder: {paths['frames_dir']}")
        print(f"Labels CSV: {paths['labels_csv']}")
        print(f"Calibration dataset path: {paths['dataset_path']}")
        print(f"Personal model path: {paths['model_path']}")

        has_model = model_exists(paths["model_path"])
        has_dataset = dataset_exists(paths["dataset_path"])
        has_calibration_files = calibration_files_exist(paths)

        action = ask_user_action(
            has_model=has_model,
            has_dataset=has_dataset,
            has_calibration_files=has_calibration_files
        )

        if action == "change_user":
            continue

        if action == "exit":
            print("\nExiting Personalized AI Posture Monitor.")
            return

        if action == "new_calibration":
            trained_successfully = run_full_calibration_and_training(paths)

            if trained_successfully:
                start_live_monitoring(paths)

            return

        if action == "recreate_dataset_and_train":
            trained_successfully = recreate_dataset_and_train(paths)

            if trained_successfully:
                start_live_monitoring(paths)

            return

        if action == "train_existing_dataset":
            trained_successfully = train_model_from_existing_dataset(paths)

            if trained_successfully:
                start_live_monitoring(paths)

            return

        if action == "live_monitoring":
            start_live_monitoring(paths)
            return


# ============================================================

if __name__ == "__main__":
    main()