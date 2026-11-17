
# 🔍 Real-Time Weapon Detection System Using YOLOv8

A computer vision application that uses a custom-trained YOLOv8 object detection model to identify and localize **Guns, Knives, and Masks** from live webcam footage.

The project combines deep learning, transfer learning, and OpenCV-based video processing to demonstrate real-time object detection using a lightweight custom-trained model.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Dataset](#-dataset)
- [Model Training](#-model-training)
- [Model Evaluation](#-model-evaluation)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [Running the Application](#-running-the-application)
- [How It Works](#-how-it-works)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)

---

## 🚀 Overview

Manual monitoring of surveillance footage can be time-consuming and may cause important events to be overlooked.

This project explores how deep learning and computer vision can assist in identifying potentially dangerous objects in video streams.

A custom YOLOv8 model was trained using an object detection dataset prepared through Roboflow. The trained model is integrated with OpenCV to process live webcam frames and display detected objects using bounding boxes, class labels, and confidence scores.

The primary objective is to develop an end-to-end object detection pipeline, from dataset preparation and model training to evaluation and real-time inference.

### Supported Classes

| Class ID | Class Name |
|---|---|
| 0 | Gun |
| 1 | Knife |
| 2 | Mask |

---

## ✨ Features

- **Custom Object Detection:** Uses a YOLOv8 model fine-tuned on a custom-labelled dataset.
- **Real-Time Webcam Processing:** Captures and processes live video using OpenCV.
- **Multi-Class Detection:** Supports gun, knife, and mask classes.
- **Bounding Box Visualization:** Localizes detected objects within video frames.
- **Confidence Scores:** Displays confidence scores for individual predictions.
- **Detection Alerts:** Displays an alert when objects belonging to trained classes are detected.
- **GPU-Based Training:** Uses Google Colab with an NVIDIA Tesla T4 GPU for model training.
- **Lightweight Model:** Uses YOLOv8 Nano for relatively efficient inference.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 (Ultralytics) | Object detection and model training |
| PyTorch | Deep learning framework |
| OpenCV | Webcam access and image processing |
| Roboflow | Dataset management and annotation |
| Google Colab | GPU-based model training |
| NumPy | Numerical and image data processing |

---

## 🏗️ System Architecture

The application follows an end-to-end computer vision pipeline:

```text
          Webcam Input
               |
               v
       Video Frame Capture
               |
               v
        OpenCV Processing
               |
               v
        YOLOv8 Detection
               |
               v
      Object Classification
               |
               v
       Bounding Box Creation
               |
               v
       Confidence Calculation
               |
               v
      Detection Visualization
               |
               v
          Live Output
```

### Main Components

**1. Video Capture**

OpenCV accesses the system webcam and continuously captures video frames.

**2. Object Detection**

Each captured frame is passed to the custom-trained YOLOv8 model.

**3. Classification and Localization**

The model predicts object classes, bounding box coordinates, and confidence scores.

**4. Visualization**

OpenCV draws bounding boxes and displays class labels and detection alerts.

---

## 📂 Dataset

The model was trained using a custom object detection dataset managed through Roboflow.

- **Dataset Version:** Version 1
- **Export Format:** YOLOv8
- **Total Images:** 2,221
- **Number of Classes:** 3

### Dataset Distribution

| Split | Images | Percentage |
|---|---:|---:|
| Training | 2,130 | 95.9% |
| Validation | 61 | 2.7% |
| Testing | 30 | 1.4% |
| **Total** | **2,221** | **100%** |

### Dataset Classes

- Gun
- Knife
- Mask

The dataset contains annotated images with bounding box coordinates and class labels in YOLO format.

The dataset is managed separately through Roboflow and is not included in this repository.

---

## 🧠 Model Training

The project uses **YOLOv8 Nano (YOLOv8n)** with pretrained weights.

Instead of training a neural network from scratch, transfer learning was used to adapt a pretrained object detection model to the custom dataset.

### Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv8 Nano |
| Initial Weights | yolov8n.pt |
| Training Framework | Ultralytics |
| Maximum Epochs | 100 |
| Actual Epochs Completed | 44 |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Hardware | NVIDIA Tesla T4 |
| Number of Classes | 3 |

Early stopping was enabled with a patience of 20 epochs.

### Training Process

1. Exported the annotated dataset from Roboflow.
2. Extracted the dataset in Google Colab.
3. Configured dataset paths using `data.yaml`.
4. Loaded pretrained YOLOv8 Nano weights.
5. Fine-tuned the model on the custom dataset.
6. Evaluated validation performance during training.
7. Saved the best-performing model checkpoint.

### Training Code

```python
from ultralytics import YOLO

# Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train on custom dataset
results = model.train(
    data="/content/weapon_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    patience=20,
    seed=42,
    project="/content/weapon_training",
    name="baseline_yolov8n",
    save=True,
    plots=True
)
```

### Model Checkpoints

After training, two important checkpoints are generated:

- `best.pt` – Best-performing model checkpoint according to validation performance.
- `last.pt` – Model checkpoint saved at the final training epoch.

The `best.pt` checkpoint is used for inference.

---

## 📊 Model Evaluation

The trained model was evaluated on a separate held-out test dataset.

### Overall Test Results

| Metric | Score |
|---|---:|
| Precision | 71.5% |
| Recall | 61.2% |
| mAP@50 | 61.5% |
| mAP@50-95 | 29.1% |

### Class-Wise Performance

| Class | Precision | Recall | mAP@50 |
|---|---:|---:|---:|
| Gun | 34.3% | 28.6% | 14.7% |
| Knife | 80.2% | 63.6% | 70.4% |
| Mask | 100% | 91.4% | 99.5% |

### Understanding the Metrics

- **Precision:** Measures how many predicted detections are correct.
- **Recall:** Measures how many actual annotated objects are successfully detected.
- **mAP@50:** Mean Average Precision at an IoU threshold of 0.50.
- **mAP@50-95:** Average detection performance across multiple IoU thresholds.

### Evaluation Observations

The initial model demonstrates different detection performance across classes.

- Knife detection performs better than gun detection.
- Gun detection has relatively low recall and mAP, indicating that the model misses many annotated gun instances.
- Mask results appear high but are based on a very small test sample.

The test dataset contains only 30 images, and one image was excluded because its annotations contained mixed detection and segmentation formats.

Therefore, these results represent an initial baseline experiment and should not be interpreted as evidence of production-ready detection performance.

---

## 📁 Project Structure

```text
weapon-detection/
│
├── weapon_detect.py
│
├── weapon_detector.pt
│
├── requirements.txt
│
└── README.md
```

### File Descriptions

| File | Description |
|---|---|
| `weapon_detect.py` | Main Python application for webcam-based detection |
| `weapon_detector.pt` | Custom-trained YOLOv8 model weights |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

The training notebook and dataset are maintained separately from the application.

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or later with compatible dependencies.
- Webcam.
- Windows, Linux, or macOS.
- Internet connection for installing dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/Shreyanshi210205/weapon_detector.git
```

Navigate into the project directory:

```bash
cd weapon-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install ultralytics opencv-python
```

### 4. Add the Trained Model

Download the trained `best.pt` checkpoint and place it in the project root.

Rename it to:

```text
weapon_detector.pt
```

The application expects this exact filename.

---

## ▶️ Running the Application

Once dependencies and model weights are installed, execute:

```bash
python weapon_detect.py
```

The application will:

1. Load the trained YOLOv8 model.
2. Initialize the webcam.
3. Capture frames continuously.
4. Run object detection on each frame.
5. Draw bounding boxes around detected objects.
6. Display class names and confidence scores.
7. Display an alert when detections occur.

Press **Q** to close the detection window.

### Confidence Threshold

The detection confidence threshold can be adjusted in `weapon_detect.py`.

```python
CONF = 0.15
```

A lower threshold allows more predictions to be displayed but can increase false positives.

The threshold should be calibrated using representative validation and test data rather than selected solely based on webcam appearance.

---

## 🔍 How It Works

The application uses the following inference workflow:

**Step 1: Model Loading**

The custom-trained YOLOv8 checkpoint is loaded using:

```python
model = YOLO("weapon_detector.pt")
```

**Step 2: Webcam Initialization**

OpenCV initializes the webcam:

```python
cap = cv2.VideoCapture(0)
```

**Step 3: Frame Processing**

Each captured frame is passed to YOLO:

```python
results = model.predict(
    source=frame,
    conf=0.15,
    imgsz=640,
    verbose=False
)
```

**Step 4: Detection Extraction**

The model returns predicted bounding boxes, class IDs, and confidence scores.

**Step 5: Visualization**

OpenCV draws bounding boxes and corresponding labels on the original video frame.

---

## ⚠️ Limitations

This project is an experimental computer vision prototype.

- Model performance depends heavily on dataset quality, class balance, lighting, camera angle, and object visibility.
- Gun detection performance is currently limited.
- The test dataset is relatively small.
- The dataset contains one annotation-format inconsistency.
- False positives and false negatives are possible.
- Confidence scores should not be interpreted as calibrated probabilities.
- The application should not be used as the sole basis for security decisions or emergency responses.

The system is intended for educational and experimental computer vision purposes, not autonomous threat assessment.

---

## 🚀 Future Improvements

Potential improvements for future iterations include:

- Increasing the number and diversity of annotated training images.
- Improving class balance, especially for gun detection.
- Cleaning inconsistent bounding-box annotations.
- Increasing validation and test dataset sizes.
- Experimenting with YOLOv8 Small and Medium models.
- Applying data augmentation to improve generalization.
- Hyperparameter tuning for better detection performance.
- Optimizing inference speed for real-time applications.
- Evaluating performance under different lighting and camera conditions.
- Comparing multiple trained model versions.

---

## ⭐ Acknowledgements

- [Ultralytics](https://docs.ultralytics.com/) – YOLO object detection framework.
- [Roboflow](https://roboflow.com/) – Dataset preparation and annotation.
- [Google Colab](https://colab.research.google.com/) – GPU-based model training.
- [OpenCV](https://opencv.org/) – Real-time computer vision and image processing.
- [PyTorch](https://pytorch.org/) – Deep learning framework.

---

**Built as a hands-on exploration of deep learning, transfer learning, and real-time object detection.**
