# 🎭 AI Deepfake Video Detection

An AI-powered deepfake video detection system that analyzes videos and predicts whether they are **REAL** or **FAKE** using deep learning.

The project combines **MobileNetV2** for visual feature extraction with **Bidirectional LSTM** and **Multi-Head Attention** for temporal analysis of video frames. A **Flask web application** provides an interactive interface for uploading videos and viewing prediction results.

---

## 🚀 Project Overview

Deepfake technology can be used to manipulate faces and visual content in videos, making it increasingly difficult to distinguish authentic media from AI-generated or manipulated content.

This project explores an end-to-end deep learning approach for deepfake video classification.

The system:

1. Accepts a video as input.
2. Samples frames from the video.
3. Detects and crops faces.
4. Extracts visual features using MobileNetV2.
5. Processes the sequence of frame features using Bidirectional LSTM layers.
6. Uses Multi-Head Attention to learn important temporal patterns.
7. Produces a REAL or FAKE prediction.
8. Displays the prediction and confidence through a Flask web application.

---

## ✨ Features

- 🎥 Video upload
- 🖱️ Drag-and-drop video upload
- 🤖 Deep learning based deepfake detection
- 👤 Face detection and cropping
- 🧠 MobileNetV2 feature extraction
- 🔄 Bidirectional LSTM temporal modeling
- 🎯 Multi-Head Attention
- 📊 Fake probability
- 📊 Real probability
- 📈 Prediction confidence
- 🖼️ Extracted frame visualization
- 🌐 Flask web interface
- 💻 Command-line prediction
- 🧠 Memory-efficient inference
- 📁 MP4, AVI, MOV, MKV and WEBM support

---

# 🏗️ System Architecture

```text
                    Input Video
                         │
                         ▼
                Video Frame Sampling
                         │
                         ▼
                   Face Detection
                         │
                         ▼
                   Face Cropping
                         │
                         ▼
                  Frame Resizing
                    224 × 224
                         │
                         ▼
                    MobileNetV2
                 Feature Extraction
                         │
                         ▼
              30 Frame Feature Sequence
                         │
                         ▼
             Bidirectional LSTM
                         │
                         ▼
             Bidirectional LSTM
                         │
                         ▼
              Multi-Head Attention
                         │
                         ▼
             Global Average Pooling
                         │
                         ▼
                  Dense Layers
                         │
                         ▼
                   Sigmoid Output
                         │
                         ▼
                    REAL / FAKE

🧠 Model Architecture
MobileNetV2
MobileNetV2 is used as the visual feature extractor.
Each processed video frame is resized to:
224 × 224 × 3

MobileNetV2 converts each frame into a:
1280-dimensional feature vector

The system processes a sequence of:
30 frames

Therefore, each video is represented by:
30 × 1280

features before temporal modeling.
Bidirectional LSTM
The extracted frame features are passed through Bidirectional LSTM layers.
The Bidirectional LSTM processes temporal information in both forward and backward directions, allowing the model to learn relationships between frames throughout the video sequence.
Multi-Head Attention
A Multi-Head Attention layer is used after the recurrent layers.
The attention mechanism helps the model focus on important temporal relationships within the sequence rather than treating every frame equally.
Classification
The temporal representation is processed using:
Global Average Pooling
        ↓
Dense 256
        ↓
Dropout
        ↓
Dense 64
        ↓
Dropout
        ↓
Sigmoid

The final sigmoid output is used to determine the classification.
📊 Dataset
The processed dataset contains 3,999 video samples.
Class	Samples
Real	2,000
Fake	1,999
Total	3,999


Each video is converted into a sequence containing:
- 30 frames
- 224 × 224 frame resolution
- 1280 MobileNetV2 features per frame
🔬 Training
The training pipeline uses:
- Training set
- Validation set
- Held-out test set
- Class weighting
- Early stopping
- Model checkpointing
- Learning-rate reduction
- Binary classification
Training configuration:
Maximum Epochs : 100
Batch Size     : 8
Sequence Length: 30
Feature Size   : 1280

The trained model is stored as:
model.keras

📈 Model Evaluation
The project uses both an internal held-out test set and a separate external evaluation set.
Internal Held-Out Test
The processed dataset was divided into:
Training   : 2800
Validation : 599
Test       : 600

At a classification threshold of 0.50, the held-out test accuracy was:
75.83%
Confusion matrix:
              Predicted
              Real  Fake

Actual Real    253    47
Actual Fake     98   202

External Evaluation
A separate evaluation was performed on 100 unseen videos:
Real : 50
Fake : 50

The final external accuracy was:
65.00%
Classification report:
              precision    recall  f1-score   support

Real            0.68       0.56      0.62        50
Fake            0.63       0.74      0.68        50

accuracy                               0.65       100
macro avg       0.66       0.65      0.65       100
weighted avg    0.66       0.65      0.65       100

Confusion matrix:
              Predicted
              Real  Fake

Actual Real     28    22
Actual Fake     13    37

The model correctly classified:
28 / 50 Real videos
37 / 50 Fake videos

The external evaluation is included to provide a more realistic assessment of performance on previously unseen videos.
⚠️ Performance Note
The internal held-out test accuracy and external evaluation accuracy represent different evaluation settings.
The internal test set is derived from the processed project dataset, while the external evaluation uses separate unseen videos.
Therefore, the project does not claim 75.83% real-world accuracy.
The external benchmark achieved:
65% accuracy on 100 unseen videos
with:
74% recall for Fake videos.
🎯 Prediction Threshold
The deployed prediction system uses:
Threshold = 0.75

The prediction logic is:
Probability >= 0.75
        ↓
      FAKE

Probability < 0.75
        ↓
      REAL

The threshold was selected using validation data.
🌐 Web Application
The project includes a Flask-based web application.
Users can:
- Upload videos
- Drag and drop videos
- Start AI analysis
- View REAL/FAKE prediction
- View prediction confidence
- View fake probability
- View real probability
- View extracted frames
The application provides a simple interface for interacting with the trained deep learning model.
💻 Installation
1. Clone the Repository
git clone https://github.com/omkar22A/deepfake-detection.git

Enter the project directory:
cd deepfake-detection

2. Create a Virtual Environment
Windows:
python -m venv venv

Activate the environment:
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

▶️ Run the Web Application
Start the Flask server:
python app.py

Open the application in your browser:
http://127.0.0.1:5000

Upload a supported video and click:
Analyze Video

The system will process the video and display the prediction.
🔎 Command-Line Prediction
Individual videos can also be analyzed from the terminal.
Run:
python predict.py video.mp4

Example:
============================================================
Prediction Result
============================================================

Prediction        : REAL
Confidence        : 66.17%
Fake Probability  : 33.83%
Real Probability  : 66.17%

============================================================

📁 Project Structure
deepfake-detection/
│
├── app.py
├── config.py
├── evaluate_folder.py
├── model.py
├── predict.py
├── preprocess.py
├── train.py
├── utils.py
│
├── model.keras
│
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── test_videos/
    ├── real/
    └── fake/

🛠️ Technologies Used
Programming
- Python
Deep Learning
- TensorFlow
- Keras
- MobileNetV2
- Bidirectional LSTM
- Multi-Head Attention
Computer Vision
- OpenCV
Data Science
- NumPy
- Scikit-learn
Web Development
- Flask
- HTML
- CSS
- JavaScript
- Bootstrap
- Font Awesome
🧩 Main Project Files
app.py
Runs the Flask web application and handles video prediction requests.
model.py
Defines the deep learning model architecture.
predict.py
Performs memory-efficient prediction on individual videos.
preprocess.py
Processes videos and creates the feature dataset.
train.py
Trains the deep learning classification model.
evaluate_folder.py
Evaluates the model using videos organized into Real and Fake folders.
utils.py
Contains shared video-processing, face-detection and feature-extraction functions.
templates/index.html
Contains the web application's user interface.
static/script.js
Handles video upload, prediction requests and result visualization.
static/style.css
Contains the application's visual styling.
💾 Memory-Efficient Inference
The prediction system was designed to work on systems with limited RAM.
Instead of processing all video frames through MobileNetV2 simultaneously, the application extracts features in small batches.
Video
  ↓
30 Frames
  ↓
Small Feature Batches
  ↓
MobileNetV2
  ↓
1280-Dimensional Features
  ↓
LSTM + Attention
  ↓
Prediction

This reduces peak memory usage during inference.
⚠️ Limitations
The current system has several limitations:
- External accuracy was 65% on the tested 100-video benchmark.
- Performance may vary depending on video quality.
- Face detection may fail under extreme poses or occlusion.
- Lighting conditions can affect predictions.
- Compression artifacts may affect model performance.
- Different deepfake generation techniques may produce different results.
- The dataset size limits generalization.
- The system is not a forensic-grade deepfake verification tool.
Therefore, predictions should be treated as an AI-based classification result rather than definitive proof that a video is authentic or manipulated.
🔮 Future Improvements
Potential future improvements include:
- Larger and more diverse training datasets
- Additional deepfake datasets
- Transformer-based temporal modeling
- More advanced face detection
- Face alignment
- Improved data augmentation
- Cross-dataset evaluation
- Better confidence calibration
- Explainable AI visualizations
- GPU acceleration
- Model ensemble techniques
- Cloud deployment
- REST API deployment
- Real-time video analysis
🎓 Academic Project
This project was developed as a:
Bachelor of Engineering — Artificial Intelligence & Data Science
The project demonstrates practical applications of:
- Computer Vision
- Deep Learning
- Transfer Learning
- Sequence Modeling
- Attention Mechanisms
- Model Evaluation
- Flask Web Development
- Machine Learning Deployment
👨‍💻 Author
Omkar Avasarkar
Artificial Intelligence & Data Science
GitHub:
https://github.com/omkar22A
Project Repository:
https://github.com/omkar22A/deepfake-detection
📜 License
This project is licensed under the MIT License.
See the LICENSE file for details.
⭐ Project Summary
This project implements an end-to-end deepfake video detection pipeline combining computer vision, transfer learning and temporal deep learning.
Video
  ↓
Face Detection
  ↓
Frame Processing
  ↓
MobileNetV2
  ↓
Temporal Feature Sequence
  ↓
Bidirectional LSTM
  ↓
Multi-Head Attention
  ↓
Classification
  ↓
REAL / FAKE
  ↓
Flask Web Application

The project covers the complete workflow from video preprocessing and feature extraction to model training, evaluation and web-based deployment.
```