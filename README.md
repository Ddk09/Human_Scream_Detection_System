# Real-Time Scream Detection System

##  Project Overview

This project is a **real-time scream detection system** that uses **machine learning and audio signal processing** to identify scream sounds from live microphone input.
When a scream is detected, the system sends alerts to a **web-based UI** in real time using **Flask and Socket.IO**.

The system is designed for **safety monitoring**, **emergency detection**, and **surveillance applications**.

---

## 🎯 Key Features

* Real-time audio capture from microphone
* MFCC-based audio feature extraction
* Random Forest classifier for scream detection
* Confidence-based decision threshold
* Live alert visualization using Flask + Socket.IO
* Audio alert + animated UI for scream events

---

## 🧠 Tech Stack

**Language:** Python
**Libraries & Tools:**

* Librosa (audio processing)
* NumPy, Pandas
* Scikit-learn (Random Forest)
* SoundDevice (live audio input)
* Flask + Flask-SocketIO
* Joblib
* Matplotlib, Seaborn
* HTML, CSS, JavaScript

---

## ⚙️ System Architecture

```
Microphone Input
        ↓
High-Pass Audio Filter
        ↓
Feature Extraction (MFCC, ZCR, Spectral Features)
        ↓
Random Forest Model
        ↓
Prediction + Confidence Score
        ↓
Flask SocketIO Backend
        ↓
Live Web UI Alert
```

---

## 📂 Project Structure

```
Scream-Detection/
│
├── scream_dataset/
│   ├── scream/
│   └── non_scream/
│
├── features_augmented.csv
├── scream_rf_model.pkl
│
├── feature_extraction.py
├── train_model.py
├── realtime_detection.py
│
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── alert.mp3

```

---

## 🧪 Feature Extraction

The following features are extracted from audio signals:

* **MFCC (13 coefficients)**
* **Zero Crossing Rate (ZCR)**
* **Spectral Centroid**
* **Spectral Rolloff**

All features are averaged over time to create a single feature vector per audio sample.

---

## 🤖 Machine Learning Model

* **Algorithm:** Random Forest Classifier
* **Estimators:** 200
* **Class Weight:** Balanced
* **Threshold:** 0.62 (confidence-based detection)

### Model Evaluation

* Accuracy score
* Confusion Matrix
* Precision, Recall, F1-score
* Feature importance visualization

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install librosa sounddevice flask flask-socketio scikit-learn joblib matplotlib seaborn rich
```

### 2️⃣ Train the Model

```bash
python train_model.py
```

### 3️⃣ Start Backend Server

```bash
python app.py
```

### 4️⃣ Run Real-Time Detection

```bash
python realtime_detection.py
```

### 5️⃣ Open UI

```
http://127.0.0.1:5000
```

---

## 🖥️ Web Interface

* Displays **SAFE** or **SCREAM** status
* Animated red alert during scream detection
* Confidence percentage shown
* Audio alert on scream events

---

## 📊 Applications

* Women safety systems
* Smart surveillance
* Emergency alert systems
* Public security monitoring
* Smart homes & offices

---

## 🔮 Future Enhancements

* Deep learning (CNN / LSTM) for higher accuracy
* Multiple sound classification
* Cloud-based alert logging
* Mobile app integration
* Noise robustness improvement

