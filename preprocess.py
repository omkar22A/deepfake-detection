"""
preprocess.py
Deepfake Video Preprocessing

- Automatically balances REAL and FAKE videos
- Extracts 30 uniformly sampled face frames
- Uses MobileNetV2 as a feature extractor
- Saves features to processed_data/X.npy and processed_data/y.npy
"""

import os
import random
import numpy as np
from tqdm import tqdm

from utils import (
    extract_frames,
    load_feature_extractor,
    frames_to_features,
)

# =====================================================
# PATHS
# =====================================================

DATASET_DIR = "dataset"

REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")

OUTPUT_DIR = "processed_data"

random.seed(42)

# =====================================================
# GET VIDEO FILES
# =====================================================

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def get_video_files(folder):

    if not os.path.exists(folder):
        return []

    return sorted([
        file
        for file in os.listdir(folder)
        if file.lower().endswith(VIDEO_EXTENSIONS)
    ])


# =====================================================
# PROCESS VIDEOS
# =====================================================

def process_video_list(video_dir, video_list, label, extractor):

    features = []
    labels = []

    processed = 0
    skipped = 0

    print(f"\nProcessing {len(video_list)} videos...")

    for video_name in tqdm(video_list):

        video_path = os.path.join(video_dir, video_name)

        frames, error = extract_frames(video_path)

        if frames is None:
            skipped += 1
            continue

        video_features = frames_to_features(
            frames,
            extractor,
        )

        features.append(video_features)
        labels.append(label)

        processed += 1

    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")

    return np.array(features), np.array(labels)


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 65)
    print("DEEPFAKE DETECTION - PREPROCESSING")
    print("=" * 65)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print("\nLoading MobileNetV2...")

    extractor = load_feature_extractor()

    print("Loaded.")

    real_videos = get_video_files(REAL_DIR)
    fake_videos = get_video_files(FAKE_DIR)

    # ====================================================
    # LIMIT DATASET SIZE (2000 REAL + 2000 FAKE = 4000)
    # ====================================================

    MAX_VIDEOS_PER_CLASS = 2000

    random.shuffle(real_videos)
    random.shuffle(fake_videos)

    real_videos = real_videos[:MAX_VIDEOS_PER_CLASS]
    fake_videos = fake_videos[:MAX_VIDEOS_PER_CLASS]

    print("\nOriginal Dataset")
    print("-------------------------")
    print("Real :", len(real_videos))
    print("Fake :", len(fake_videos))

    # ================================================
    # Dataset Used (balancing removed)
    # ================================================

    print("\nDataset Used")
    print(f"Real : {len(real_videos)}")
    print(f"Fake : {len(fake_videos)}")

    # ================================================
    # Extract Features
    # ================================================

    real_X, real_y = process_video_list(
        REAL_DIR,
        real_videos,
        0,
        extractor,
    )

    fake_X, fake_y = process_video_list(
        FAKE_DIR,
        fake_videos,
        1,
        extractor,
    )

    if len(real_X) == 0 or len(fake_X) == 0:
        print("\nERROR : No videos processed.")
        return

    X = np.concatenate(
        [real_X, fake_X],
        axis=0,
    )

    y = np.concatenate(
        [real_y, fake_y],
        axis=0,
    )

    # ================================================
    # Shuffle Dataset
    # ================================================

    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    print("\nFinal Dataset")
    print("-------------------------")
    print("X Shape :", X.shape)
    print("y Shape :", y.shape)

    np.save(
        os.path.join(OUTPUT_DIR, "X.npy"),
        X,
    )

    np.save(
        os.path.join(OUTPUT_DIR, "y.npy"),
        y,
    )

    print("\nSaved Successfully")
    print("-------------------------")
    print("processed_data/X.npy")
    print("processed_data/y.npy")

    print("\nNext Step")
    print("python train.py")


if __name__ == "__main__":
    main()