"""
utils.py
Shared utility functions for preprocessing, training and prediction.
"""

import os
import cv2
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
)

# ===========================================================
# CONFIGURATION
# ===========================================================

IMG_SIZE = 224

SEQUENCE_LENGTH = 30          # increased from 15

MODEL_PATH = "model.keras"

ALLOWED_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm",
}

# ===========================================================
# FACE DETECTOR
# ===========================================================

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ===========================================================
# FILE CHECK
# ===========================================================


def allowed_file(filename):

    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ===========================================================
# FACE CROP
# ===========================================================

def crop_face(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    h, w = gray.shape

    if len(faces) == 0:

        side = min(h, w)

        x = (w - side) // 2

        y = (h - side) // 2

        crop = rgb[y:y + side, x:x + side]

    else:

        x, y, fw, fh = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        margin = int(fw * 0.25)

        x1 = max(0, x - margin)

        y1 = max(0, y - margin)

        x2 = min(w, x + fw + margin)

        y2 = min(h, y + fh + margin)

        crop = rgb[y1:y2, x1:x2]

    crop = cv2.resize(
        crop,
        (IMG_SIZE, IMG_SIZE)
    )

    return crop


# ===========================================================
# FRAME EXTRACTION
# ===========================================================

def extract_frames(video_path):

    if not os.path.exists(video_path):

        return None, "Video not found"

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames < SEQUENCE_LENGTH:

        cap.release()

        return None, "Video too short"

    frame_numbers = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
        dtype=int,
    )

    frames = []

    for frame_no in frame_numbers:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_no),
        )

        success, frame = cap.read()

        if not success:

            continue

        face = crop_face(frame)

        frames.append(face)

    cap.release()

    if len(frames) != SEQUENCE_LENGTH:

        return None, "Frame extraction failed"

    return np.array(frames), None


# ===========================================================
# FEATURE EXTRACTOR
# ===========================================================

def load_feature_extractor():

    model = MobileNetV2(

        weights="imagenet",

        include_top=False,

        pooling="avg",

        input_shape=(IMG_SIZE, IMG_SIZE, 3),

    )

    return model


# ===========================================================
# FEATURE EXTRACTION
# ===========================================================

def frames_to_features(

    frames,

    feature_extractor,

):

    frames = preprocess_input(

        frames.astype(np.float32)

    )

    features = feature_extractor.predict(

        frames,

        verbose=0,

    )

    return features