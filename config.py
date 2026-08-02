import os

# ==========================
# Dataset
# ==========================

DATASET_DIR = "dataset"
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")

# ==========================
# Processed Data
# ==========================

PROCESSED_DIR = "processed_data"

# ==========================
# Model
# ==========================

MODEL_PATH = "model.keras"

# ==========================
# Image
# ==========================

IMG_SIZE = 224

# Increase from 15
SEQUENCE_LENGTH = 30

# ==========================
# Training
# ==========================

EPOCHS = 50

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

LSTM_UNITS = 256

DROPOUT = 0.4