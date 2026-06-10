# 🩺 Skin Disease Detection using Vision Transformer (ViT)

An advanced deep learning framework for **automated skin disease classification** using **Vision Transformers (ViT)** and modern training strategies such as **contrastive fine-tuning**, **hard sample mining**, and **test-time augmentation (TTA)**.

## 📌 Project Overview

Skin diseases are among the most common medical conditions worldwide. Early and accurate diagnosis plays a crucial role in effective treatment. This project proposes an intelligent deep learning framework capable of classifying dermoscopic skin images into multiple disease categories.

The project explores multiple architectures and optimization strategies to improve classification performance and model robustness.

---

## 🚀 Features

✅ Vision Transformer (ViT)-based classification
✅ Contrastive fine-tuning for better feature learning
✅ Hard sample mining for improved robustness
✅ Test Time Augmentation (TTA) for enhanced predictions
✅ Attention visualization for explainability (XAI)
✅ Embedding separability analysis using t-SNE
✅ Model comparison and evaluation pipeline
✅ Confusion matrix and prediction visualizations

---

## 🧠 Tech Stack

* **Python**
* **TensorFlow / Keras**
* **Vision Transformer (ViT)**
* **NumPy**
* **Matplotlib**
* **scikit-learn**
* **OpenCV**

---

## 📂 Project Structure

```text
skin-disease-detection-vit/
│── src/                     # Training & evaluation scripts
│── results/                 # Predictions, confusion matrix, plots
│── models/                  # Model details
│── requirements.txt
│── .gitignore
│── README.md
```

---

## 📊 Methodology

The framework includes:

### 1. Data Preparation

* Dataset preprocessing
* Label preparation
* Balanced training strategies

### 2. Model Training

* Vision Transformer (ViT)
* Fine-tuning strategies
* Focal loss experimentation
* Micro fine-tuning

### 3. Performance Enhancement

* Hard sample mining
* Contrastive learning
* Test Time Augmentation (TTA)

### 4. Explainability & Analysis

* Attention map visualization
* Embedding analysis
* t-SNE feature separability

---

## 📈 Results

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

### Model Architecture

![Model Architecture](results/model_architecture.png)

### Sample Predictions

| Prediction 1                  | Prediction 2                  | Prediction 3                  |
| ----------------------------- | ----------------------------- | ----------------------------- |
| ![](results/prediction_1.png) | ![](results/prediction_2.png) | ![](results/prediction_3.png) |

---

## 📦 Dataset

This project is trained using **ISIC (International Skin Imaging Collaboration)** dermoscopic image datasets.

Dataset is not included in this repository due to size limitations.

---

## ⚠️ Model Weights

Trained model weights are excluded due to GitHub file size limitations.

---

## 🔮 Future Improvements

* Deploy as a web application
* Real-time skin disease prediction
* Mobile integration
* Better explainability for medical trust

---

## 👨‍💻 Author

**Sahaj Dang**
B.Tech Data Science | Machine Learning & Software Development Enthusiast

