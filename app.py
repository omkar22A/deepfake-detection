import os
import cv2
import base64
import numpy as np

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model

from utils import (
    MODEL_PATH,
    allowed_file,
    extract_frames,
    load_feature_extractor,
    frames_to_features,
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

print("Loading MobileNetV2...")
feature_extractor = load_feature_extractor()

print("Loading trained model...")
model = load_model(MODEL_PATH)

print("System Ready.")


def encode_frames(frames):

    previews = []

    for frame in frames:

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode(".jpg", frame)

        previews.append(
            base64.b64encode(buffer).decode("utf-8")
        )

    return previews


def predict_video(video_path):

    frames, error = extract_frames(video_path)

    if frames is None:

        return {
            "success": False,
            "error": error
        }

    features = frames_to_features(
        frames,
        feature_extractor
    )

    features = np.expand_dims(features, axis=0)

    probability = float(
        model.predict(features, verbose=0)[0][0]
    )

    fake_probability = probability * 100
    real_probability = (1 - probability) * 100

    label = "FAKE" if probability >= 0.5 else "REAL"

    confidence = max(
        fake_probability,
        real_probability
    )

    previews = encode_frames(frames[:6])

    return {

        "success": True,

        "label": label,

        "confidence": round(confidence,2),

        "fake_probability": round(fake_probability,2),

        "real_probability": round(real_probability,2),

        "frames": previews

    }


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():

    if "file" not in request.files:

        return jsonify({

            "success": False,

            "error":"No file uploaded"

        })

    file = request.files["file"]

    if file.filename == "":

        return jsonify({

            "success": False,

            "error":"No file selected"

        })

    if not allowed_file(file.filename):

        return jsonify({

            "success":False,

            "error":"Unsupported format"

        })

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    result = predict_video(filepath)

    if os.path.exists(filepath):

        os.remove(filepath)

    return jsonify(result)


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )