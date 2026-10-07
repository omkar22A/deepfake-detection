"""
Tune the Deepfake Detection decision threshold
using ONLY the validation split.

Memory-efficient version:
uses indices instead of copying X.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

MODEL_PATH = "model.keras"

print("=" * 70)
print("DEEPFAKE DETECTION - THRESHOLD TUNING")
print("=" * 70)

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("\nLoading processed data...")

X = np.load(
    "processed_data/X.npy",
    mmap_mode="r"
)

y = np.load("processed_data/y.npy")

print(f"Samples : {len(X)}")
print(f"Real    : {np.sum(y == 0)}")
print(f"Fake    : {np.sum(y == 1)}")

# --------------------------------------------------
# Reproduce EXACT same split used in train.py
# --------------------------------------------------

indices = np.arange(len(y))

train_temp_idx, test_idx = train_test_split(
    indices,
    test_size=0.15,
    random_state=42,
    stratify=y,
)

train_idx, val_idx = train_test_split(
    train_temp_idx,
    test_size=0.176,
    random_state=42,
    stratify=y[train_temp_idx],
)

print("\nDataset Split")
print("-------------------------")
print("Train      :", len(train_idx))
print("Validation :", len(val_idx))
print("Test       :", len(test_idx))

# --------------------------------------------------
# Load trained model
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} not found."
    )

print("\nLoading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")

# --------------------------------------------------
# Validation predictions
# --------------------------------------------------

print("\nGenerating validation predictions...")

val_probabilities = model.predict(
    X[val_idx],
    verbose=1
).flatten()

# --------------------------------------------------
# Search thresholds
# --------------------------------------------------

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

results = []

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y[val_idx],
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y[val_idx],
        predictions
    )

    f1_macro = f1_score(
        y[val_idx],
        predictions,
        average="macro"
    )

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": f1_macro,
    })

# --------------------------------------------------
# Select threshold
# --------------------------------------------------

best = max(
    results,
    key=lambda x: x["balanced_accuracy"]
)

best_threshold = best["threshold"]

print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"\nThreshold          : {best_threshold:.2f}"
)

print(
    f"Validation Accuracy: "
    f"{best['accuracy'] * 100:.2f}%"
)

print(
    f"Balanced Accuracy  : "
    f"{best['balanced_accuracy'] * 100:.2f}%"
)

print(
    f"Macro F1           : "
    f"{best['f1_macro'] * 100:.2f}%"
)

# --------------------------------------------------
# Top thresholds
# --------------------------------------------------

print("\nTop 10 Thresholds")
print("-" * 70)

top_results = sorted(
    results,
    key=lambda x: x["balanced_accuracy"],
    reverse=True
)[:10]

for result in top_results:

    print(
        f"Threshold: {result['threshold']:.2f} | "
        f"Accuracy: {result['accuracy'] * 100:.2f}% | "
        f"Balanced Accuracy: "
        f"{result['balanced_accuracy'] * 100:.2f}% | "
        f"Macro F1: {result['f1_macro'] * 100:.2f}%"
    )

# --------------------------------------------------
# Save threshold
# --------------------------------------------------

with open("threshold.txt", "w") as file:
    file.write(f"{best_threshold:.2f}")

print(
    "\nSaved final threshold to: threshold.txt"
)

# --------------------------------------------------
# Test set evaluation
# Threshold was NOT selected using test data.
# --------------------------------------------------

print("\nGenerating test predictions...")

test_probabilities = model.predict(
    X[test_idx],
    verbose=1
).flatten()

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_accuracy = accuracy_score(
    y[test_idx],
    test_predictions
)

print("\n" + "=" * 70)
print("TEST RESULTS WITH TUNED THRESHOLD")
print("=" * 70)

print(
    f"\nAccuracy : {test_accuracy * 100:.2f}%"
)

print("\nClassification Report\n")

print(
    classification_report(
        y[test_idx],
        test_predictions,
        target_names=["Real", "Fake"]
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y[test_idx],
        test_predictions
    )
)

print("\n" + "=" * 70)
print("THRESHOLD TUNING COMPLETE")
print("=" * 70)