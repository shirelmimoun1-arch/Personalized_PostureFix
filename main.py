# ============================================================
# MAIN – PERSONALIZED AI POSTURE MONITOR
# Full pipeline + live monitoring
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
    while True:
        user_id = input("Enter user name / ID: ").strip()

        if user_id:
            break

        print("Enter user name / ID:")

    return user_id.strip().replace(" ", "_").lower()


def get_user_paths(user_id):
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
    return os.path.exists(model_path)


def dataset_exists(dataset_path):
    return os.path.exists(dataset_path)


def calibration_files_exist(paths):
    return (
        os.path.exists(paths["labels_csv"])
        and os.path.exists(paths["frames_dir"])
        and len(os.listdir(paths["frames_dir"])) > 0
    )


def ask_user_action(has_model, has_dataset, has_calibration_files):
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
    print("\nStarting automatic calibration.")

    run_calibration(
        frames_dir=paths["frames_dir"],
        labels_csv_path=paths["labels_csv"],
        target_fps=5
    )

    create_dataset_from_existing_calibration(paths)
    return train_model_from_existing_dataset(paths)


def recreate_dataset_and_train(paths):
    if not calibration_files_exist(paths):
        print("\nNo existing frames/labels found.")
        print("Please run new calibration first.")
        return False

    create_dataset_from_existing_calibration(paths)
    return train_model_from_existing_dataset(paths)


def start_live_monitoring(paths):
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