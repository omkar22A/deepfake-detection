"""
Professional Deepfake Video Prediction Script
"""

import os
import sys
import numpy as np
from tensorflow.keras.models import load_model

from utils import (
    MODEL_PATH,
    extract_frames,
    load_feature_extractor,
    frames_to_features,
)

THRESHOLD = 0.5


class DeepfakePredictor:

    def __init__(self):

        print("Loading MobileNetV2...")
        self.feature_extractor = load_feature_extractor()

        print("Loading trained model...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} not found.")

        self.model = load_model(MODEL_PATH)

        print("Model Loaded Successfully.\n")

    def predict(self, video_path):

        frames, error = extract_frames(video_path)

        if frames is None:
            return {
                "success": False,
                "error": error
            }

        features = frames_to_features(
            frames,
            self.feature_extractor
        )

        features = np.expand_dims(features, axis=0)

        probability = float(
            self.model.predict(features, verbose=0)[0][0]
        )

        label = "FAKE" if probability >= THRESHOLD else "REAL"

        confidence = max(probability, 1 - probability)

        return {

            "success": True,

            "label": label,

            "confidence": confidence * 100,

            "fake_probability": probability * 100,

            "real_probability": (1 - probability) * 100

        }


def main():

    if len(sys.argv) != 2:

        print()

        print("Usage")

        print("python predict.py video.mp4")

        return

    video_path = sys.argv[1]

    if not os.path.exists(video_path):

        print("Video not found.")

        return

    detector = DeepfakePredictor()

    result = detector.predict(video_path)

    if not result["success"]:

        print(result["error"])

        return

    print("=" * 60)

    print("Prediction Result")

    print("=" * 60)

    print()

    print(f"Prediction        : {result['label']}")

    print(f"Confidence        : {result['confidence']:.2f}%")

    print(f"Fake Probability  : {result['fake_probability']:.2f}%")

    print(f"Real Probability  : {result['real_probability']:.2f}%")

    print()

    print("=" * 60)


if __name__ == "__main__":
    main()