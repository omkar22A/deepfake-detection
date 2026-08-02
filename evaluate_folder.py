import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from predict import DeepfakePredictor

REAL_DIR = "test_videos/real"
FAKE_DIR = "test_videos/fake"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def get_videos(folder):
    videos = []
    for file in os.listdir(folder):
        if file.lower().endswith(VIDEO_EXTENSIONS):
            videos.append(file)
    return sorted(videos)


def evaluate_folder(detector, folder, true_label):

    y_true = []
    y_pred = []

    files = get_videos(folder)

    print(f"\nTesting {len(files)} videos from {folder}\n")

    for i, file in enumerate(files, 1):

        path = os.path.join(folder, file)

        result = detector.predict(path)

        if not result["success"]:
            print(f"Skipped : {file}")
            continue

        pred = 1 if result["label"] == "FAKE" else 0

        y_true.append(true_label)
        y_pred.append(pred)

        print(
            f"[{i}/{len(files)}] "
            f"{file} -> {result['label']} "
            f"({result['confidence']:.2f}%)"
        )

    return y_true, y_pred


def main():

    detector = DeepfakePredictor()

    real_true, real_pred = evaluate_folder(
        detector,
        REAL_DIR,
        0,
    )

    fake_true, fake_pred = evaluate_folder(
        detector,
        FAKE_DIR,
        1,
    )

    y_true = real_true + fake_true
    y_pred = real_pred + fake_pred

    print("\n" + "=" * 70)

    print("FINAL RESULTS")

    print("=" * 70)

    print("\nOverall Accuracy : {:.2f}%".format(
        accuracy_score(y_true, y_pred) * 100
    ))

    print("\nClassification Report\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Real", "Fake"],
        )
    )

    print("\nConfusion Matrix\n")

    print(
        confusion_matrix(
            y_true,
            y_pred,
        )
    )


if __name__ == "__main__":
    main()