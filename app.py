"""
Deepfake Detection Flask Web Application
Memory-safe inference for 8 GB RAM systems
"""

import os
import gc
import cv2
import base64
import numpy as np

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from utils import (
    MODEL_PATH,
    allowed_file,
    extract_frames,
    load_feature_extractor,
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# ============================================================
# MODEL CONFIGURATION
# ============================================================

THRESHOLD = 0.75

FEATURE_BATCH_SIZE = 2


print("Loading MobileNetV2...")

feature_extractor = load_feature_extractor()


print("Loading trained model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"{MODEL_PATH} not found."
    )

model = load_model(MODEL_PATH)


print("System Ready.")


# ============================================================
# FRAME ENCODING
# ============================================================

def encode_frames(frames):

    previews = []

    for frame in frames:

        frame_bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        success, buffer = cv2.imencode(
            ".jpg",
            frame_bgr
        )

        if success:

            previews.append(
                base64.b64encode(
                    buffer
                ).decode("utf-8")
            )

    return previews


# ============================================================
# MEMORY-SAFE FEATURE EXTRACTION
# ============================================================

def extract_features_memory_safe(frames):

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

        features = feature_extractor.predict(
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


# ============================================================
# VIDEO PREDICTION
# ============================================================

def predict_video(video_path):

    try:

        # -----------------------------------------
        # Extract frames
        # -----------------------------------------

        frames, error = extract_frames(
            video_path
        )

        if frames is None:

            return {
                "success": False,
                "error": error
            }


        # -----------------------------------------
        # Save preview frames before cleanup
        # -----------------------------------------

        preview_frames = frames[:6].copy()


        # -----------------------------------------
        # MobileNetV2 feature extraction
        # -----------------------------------------

        features = extract_features_memory_safe(
            frames
        )

        del frames

        gc.collect()


        # -----------------------------------------
        # LSTM / Attention model prediction
        # -----------------------------------------

        sequence = np.expand_dims(
            features,
            axis=0
        )

        del features

        gc.collect()


        probability = float(
            model.predict(
                sequence,
                verbose=0
            )[0][0]
        )


        del sequence

        gc.collect()


        # -----------------------------------------
        # Final classification
        # -----------------------------------------

        fake_probability = probability * 100

        real_probability = (
            1 - probability
        ) * 100


        label = (
            "FAKE"
            if probability >= THRESHOLD
            else "REAL"
        )


        confidence = max(
            fake_probability,
            real_probability
        )


        # -----------------------------------------
        # Encode preview frames
        # -----------------------------------------

        previews = encode_frames(
            preview_frames
        )

        del preview_frames

        gc.collect()


        # -----------------------------------------
        # Return result
        # -----------------------------------------

        return {

            "success": True,

            "label": label,

            "confidence": round(
                confidence,
                2
            ),

            "fake_probability": round(
                fake_probability,
                2
            ),

            "real_probability": round(
                real_probability,
                2
            ),

            "frames": previews

        }


    except Exception as e:

        gc.collect()

        print("\nPrediction Error:")
        print(str(e))

        return {

            "success": False,

            "error": (
                "Prediction failed: "
                + str(e)
            )

        }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        # -----------------------------------------
        # Check uploaded file
        # -----------------------------------------

        if "file" not in request.files:

            return jsonify({

                "success": False,

                "error": "No file uploaded"

            }), 400


        file = request.files["file"]


        if file.filename == "":

            return jsonify({

                "success": False,

                "error": "No file selected"

            }), 400


        # -----------------------------------------
        # Validate extension
        # -----------------------------------------

        if not allowed_file(
            file.filename
        ):

            return jsonify({

                "success": False,

                "error": "Unsupported video format"

            }), 400


        # -----------------------------------------
        # Save uploaded video
        # -----------------------------------------

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)


        # -----------------------------------------
        # Run prediction
        # -----------------------------------------

        result = predict_video(
            filepath
        )


        # -----------------------------------------
        # Delete uploaded file
        # -----------------------------------------

        if os.path.exists(filepath):

            os.remove(filepath)


        return jsonify(result)


    except Exception as e:

        print("\nAPI Error:")
        print(str(e))

        return jsonify({

            "success": False,

            "error": (
                "Server error: "
                + str(e)
            )

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )