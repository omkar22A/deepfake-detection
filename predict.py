"""
Professional Deepfake Video Prediction
Memory-safe inference for low-RAM systems
"""

import os
import sys
import gc
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from utils import (
    MODEL_PATH,
    extract_frames,
    load_feature_extractor,
)

# Final threshold selected using validation data
THRESHOLD = 0.75

# Small batch keeps RAM usage low on 8 GB systems
FEATURE_BATCH_SIZE = 2


class DeepfakePredictor:

    def __init__(self):

        print("Loading MobileNetV2...")
        self.feature_extractor = load_feature_extractor()

        print("Loading trained model...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"{MODEL_PATH} not found."
            )

        self.model = load_model(MODEL_PATH)

        print("Model Loaded Successfully.\n")

    def extract_features_memory_safe(self, frames):

        frames = frames.astype(
            np.float32,
            copy=False
        )

        frames = preprocess_input(frames)

        feature_batches = []

        for start in range(
            0,
            len(frames),
            FEATURE_BATCH_SIZE
        ):

            batch = frames[
                start:start + FEATURE_BATCH_SIZE
            ]

            features = self.feature_extractor.predict(
                batch,
                batch_size=FEATURE_BATCH_SIZE,
                verbose=0
            )

            feature_batches.append(features)

            del batch
            del features

            gc.collect()

        features = np.concatenate(
            feature_batches,
            axis=0
        )

        del feature_batches
        del frames

        gc.collect()

        return features

    def predict(self, video_path):

        try:

            frames, error = extract_frames(video_path)

            if frames is None:
                return {
                    "success": False,
                    "error": error
                }

            features = self.extract_features_memory_safe(
                frames
            )

            del frames
            gc.collect()

            sequence = np.expand_dims(
                features,
                axis=0
            )

            del features
            gc.collect()

            probability = float(
                self.model.predict(
                    sequence,
                    verbose=0
                )[0][0]
            )

            del sequence
            gc.collect()

            label = (
                "FAKE"
                if probability >= THRESHOLD
                else "REAL"
            )

            confidence = max(
                probability,
                1 - probability
            )

            return {
                "success": True,
                "label": label,
                "confidence": confidence * 100,
                "fake_probability": probability * 100,
                "real_probability": (
                    1 - probability
                ) * 100
            }

        except Exception as e:

            gc.collect()

            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}"
            }


def main():

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print("python predict.py video.mp4")
        print()

        return

    video_path = sys.argv[1]

    if not os.path.exists(video_path):

        print("Video not found.")
        return

    detector = DeepfakePredictor()

    result = detector.predict(video_path)

    if not result["success"]:

        print()
        print("ERROR:")
        print(result["error"])
        return

    print()
    print("=" * 60)
    print("Prediction Result")
    print("=" * 60)
    print()

    print(
        f"Prediction        : "
        f"{result['label']}"
    )

    print(
        f"Confidence        : "
        f"{result['confidence']:.2f}%"
    )

    print(
        f"Fake Probability  : "
        f"{result['fake_probability']:.2f}%"
    )

    print(
        f"Real Probability  : "
        f"{result['real_probability']:.2f}%"
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()