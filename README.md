# 🎭 Deepfake Video Detection

A deep learning–based web application for detecting whether a video is **REAL or FAKE** using a hybrid **MobileNetV2 + LSTM** architecture.

The project combines spatial feature extraction from video frames with temporal sequence modelling to analyze patterns across a video and produce a final deepfake prediction with confidence and probability scores.

---

## 🚀 Project Overview

Deepfake technology can generate highly realistic manipulated videos, creating challenges for digital media verification and online safety.

This project aims to provide an accessible deepfake detection system where users can upload a video through a web interface and receive:

- **REAL / FAKE classification**
- Prediction confidence
- Fake probability
- Real probability
- Selected video-frame previews
- Memory-efficient inference for systems with limited RAM

---

## 🧠 Model Architecture

The system uses a hybrid **CNN + LSTM** approach.

```text
                 Input Video
                      │
                      ▼
              Frame Extraction
                      │
                      ▼
                Video Frames
                      │
                      ▼
               MobileNetV2
            Spatial Feature Extraction
                      │
                      ▼
             Sequential Features
                      │
                      ▼
                    LSTM
             Temporal Modelling
                      │
                      ▼
              Binary Prediction
                      │
                      ▼
          Threshold-based Decision
             ┌────────┴────────┐
             ▼                 ▼
           REAL               FAKE
```

### MobileNetV2

MobileNetV2 is used as the convolutional feature extractor.

It converts individual video frames into compact visual feature representations while keeping computational requirements relatively low.

### LSTM

The extracted frame features are treated as a temporal sequence and passed to an LSTM-based model.

This allows the system to learn patterns that occur across multiple frames rather than relying on a single image.

### Sequence Configuration

The project uses:

- **Image size:** 224 × 224
- **Sequence length:** 15 frames
- **Feature extractor:** MobileNetV2
- **Temporal model:** LSTM
- **Task:** Binary classification

---

## 🎯 Decision Threshold

Instead of blindly using the default `0.50` classification threshold, the project includes a dedicated threshold-tuning procedure.

`tune_threshold.py`:

1. Reproduces the validation split used during training.
2. Generates validation predictions.
3. Tests thresholds from `0.10` to `0.90`.
4. Evaluates accuracy, balanced accuracy and macro F1.
5. Selects the threshold with the best validation balanced accuracy.
6. Saves the selected threshold to `threshold.txt`.
7. Evaluates the selected threshold on the held-out test set.

The current tuned threshold is:

```text
0.75
```

The threshold is selected using validation data rather than test data, helping prevent test-set leakage during threshold selection.

---

## 💻 Web Application

The project includes a Flask-based web interface.

### Main functionality

- Video upload
- File validation
- Frame extraction
- Memory-efficient feature extraction
- Deepfake prediction
- Confidence calculation
- Real/Fake probability display
- Video-frame previews
- Automatic cleanup of uploaded videos

---

## 🛠️ Technologies Used

### Programming

- Python

### Machine Learning / Deep Learning

- TensorFlow
- Keras
- MobileNetV2
- LSTM
- Scikit-learn

### Computer Vision

- OpenCV
- NumPy

### Web Development

- Flask
- HTML
- CSS
- JavaScript

### Data Processing

- NumPy
- Scikit-learn

---

## 📁 Project Structure

```text
deepfake-detection/
│
├── app.py
├── predict.py
├── train.py
├── tune_threshold.py
├── utils.py
├── threshold.txt
├── model.keras
│
├── templates/
│   └── index.html
│
├── static/
│   └── script.js
│
├── processed_data/
│   ├── X.npy
│   └── y.npy
│
├── dataset/
│
├── test_videos/
│
└── uploads/
```

> Large datasets, generated files, uploaded videos and the Python virtual environment are excluded from version control through `.gitignore`.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/omkar22A/deepfake-detection.git
cd deepfake-detection
```

### 2. Create a virtual environment

Python 3.11 is recommended for compatibility with the project's TensorFlow/Keras environment.

```bash
python -m venv venv
```

### 3. Activate the environment

### Windows

```powershell
venv\Scripts\activate
```

### 4. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the required Python packages according to the project's environment.

---

## ▶️ Running the Web Application

Start the Flask application:

```bash
python app.py
```

The application will be available locally at:

```text
http://127.0.0.1:5000
```

Open the address in a browser and upload a supported video.

---

## 🔬 Command-Line Prediction

The prediction pipeline can also be executed directly:

```bash
python predict.py video.mp4
```

Example output:

```text
============================================================
Prediction Result
============================================================

Prediction        : FAKE
Confidence        : XX.XX%
Fake Probability  : XX.XX%
Real Probability  : XX.XX%

============================================================
```

---

## 🧪 Threshold Tuning

To reproduce threshold selection:

```bash
python tune_threshold.py
```

The script uses the validation split to determine the best classification threshold and writes the selected value to:

```text
threshold.txt
```

This keeps threshold optimization separate from the main inference pipeline.

---

## 🧠 Memory-Efficient Design

The application was designed with lower-RAM systems in mind.

Key optimizations include:

- Small feature-extraction batch size
- Batch-wise MobileNetV2 inference
- Explicit deletion of large NumPy arrays
- Python garbage collection
- Memory-mapped processed datasets during threshold tuning
- Avoiding unnecessary copies of large frame arrays

The inference batch size is currently:

```python
FEATURE_BATCH_SIZE = 2
```

This helps reduce peak RAM usage during video processing.

---

## 📊 Evaluation

The threshold-tuning pipeline evaluates:

- Accuracy
- Balanced Accuracy
- Macro F1
- Classification Report
- Confusion Matrix

The threshold is optimized on the validation set and then evaluated on the held-out test set.

### Important

The project intentionally does **not** claim a single headline accuracy number here without a reproducible evaluation result being documented.

This avoids presenting a potentially misleading performance figure.

---

## 🔐 File Handling

Uploaded videos are:

1. Validated for supported file types.
2. Saved temporarily for processing.
3. Passed through the prediction pipeline.
4. Removed after prediction.

The application also limits upload size to **500 MB**.

---

## ⚠️ Limitations

Deepfake detection is an evolving research problem.

The model's prediction can be affected by:

- Video quality
- Compression
- Lighting
- Face visibility
- Unusual camera angles
- Dataset bias
- Previously unseen manipulation techniques

Therefore, the output should be treated as a **model prediction rather than definitive proof** that a video is authentic or manipulated.

---

## 🔮 Future Improvements

Potential improvements include:

- Larger and more diverse training datasets
- Face detection and alignment improvements
- Advanced temporal architectures
- Transformer-based video modelling
- Better calibration of prediction confidence
- Explainable AI / visual attention maps
- GPU-accelerated inference
- Cloud deployment
- REST API deployment
- Automated evaluation dashboards

---

## 📌 Key Learning Outcomes

This project provided practical experience with:

- Deep learning model development
- CNN feature extraction
- Transfer learning
- LSTM-based sequence modelling
- Video preprocessing
- Computer vision with OpenCV
- Model evaluation
- Threshold optimization
- Flask application development
- Frontend-backend integration
- Memory optimization
- Git and GitHub version control

---

## 👨‍💻 Author

**Omkar Avasarkar**

AI & Data Science Engineer

Interested in:

- Data Science
- Machine Learning
- Data Analytics
- Artificial Intelligence
- Python Development

---

## ⭐ Project

If you find the project useful or interesting, consider giving the repository a ⭐ on GitHub.