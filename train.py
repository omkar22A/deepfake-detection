"""
train.py
Train the Deepfake Detection model
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from model import build_model

# ==========================
# CONFIGURATION
# ==========================

MODEL_PATH = "model.keras"

EPOCHS = 100
BATCH_SIZE = 8

# ==========================

print("=" * 70)
print("DEEPFAKE DETECTION - MODEL TRAINING")
print("=" * 70)

# ==========================
# Load Dataset
# ==========================

print("\nLoading processed data...")

X = np.load("processed_data/X.npy")
y = np.load("processed_data/y.npy")

print(f"Samples : {len(X)}")
print(f"Real    : {np.sum(y==0)}")
print(f"Fake    : {np.sum(y==1)}")

# ==========================
# Split Dataset
# ==========================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.176,
    random_state=42,
    stratify=y_temp,
)

print("\nDataset Split")
print("-------------------------")
print("Train      :", len(X_train))
print("Validation :", len(X_val))
print("Test       :", len(X_test))

# ==========================
# Class Weights
# ==========================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)

class_weights = {
    0: weights[0],
    1: weights[1],
}

print("\nClass Weights")
print(class_weights)

# ==========================
# Build Model
# ==========================

model = build_model(
    sequence_length=X.shape[1],
    feature_dim=X.shape[2],
)

model.summary()

# ==========================
# Callbacks
# ==========================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1,
)

# ==========================
# Training
# ==========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr,
    ],
    verbose=1,
)

# ==========================
# Evaluation
# ==========================

print("\nEvaluating...\n")

probabilities = model.predict(X_test)

predictions = (probabilities > 0.50).astype(int).flatten()

accuracy = accuracy_score(
    y_test,
    predictions,
)

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Real", "Fake"],
    )
)

cm = confusion_matrix(
    y_test,
    predictions,
)

print("\nConfusion Matrix\n")

print(cm)

# ==========================
# Save Confusion Matrix
# ==========================

plt.figure(figsize=(7,6))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=["Real","Fake"],
    yticklabels=["Real","Fake"],
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
)

plt.close()

# ==========================
# Training Curves
# ==========================

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train","Validation"])

plt.grid()

plt.subplot(1,2,2)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train","Validation"])

plt.grid()

plt.tight_layout()

plt.savefig(
    "training_curves.png",
    dpi=300,
)

plt.close()

print("\nSaved")
print("training_curves.png")
print("confusion_matrix.png")

print(f"\nBest model saved as {MODEL_PATH}")

print("\nTraining Complete.")