# Breast Ultrasound Image Classification 🎗️

This project implements and compares two different approaches for classifying Breast Ultrasound Images (BUSI) into three categories: **Benign, Malignant, and Normal**.

## 📊 Project Overview

The goal of this project is to automate the diagnosis of breast cancer using ultrasound imagery. Two methodologies were implemented and compared:
1.  **Deep Learning (Transfer Learning):** Using **EfficientNet-B0** pre-trained on ImageNet.
2.  **Classical Machine Learning:** Using **Random Forest** with PCA dimensionality reduction.

## 📂 Dataset
The project utilizes the [Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).
- **Preprocessing:** Images resized to 224x224 (DL) and 64x64 (ML).
- **Augmentation:** A TensorFlow script generates 4,000 augmented images per class using rotation, flipping, and cropping to balance the dataset.

## 🧠 Methodology

### 1. Data Augmentation
Due to the limited size of medical datasets, we expanded the dataset using `TensorFlow` image operations.
- **Script:** `notebooks/01_data_augmentation.py`
- **Techniques:** Random flip (up/down, left/right), Random crop, Rotation.

### 2. EfficientNet-B0 (PyTorch)
We fine-tuned a pre-trained EfficientNet-B0 model.
- **Optimizer:** Adam (lr=1e-4)
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 25
- **Result:** ~96% Accuracy

### 3. Random Forest (Scikit-Learn)
We flattened images and reduced dimensions using PCA (100 components) before training a Random Forest classifier.
- **Estimators:** 100
- **Class Weights:** Balanced
- **Result:** ~58% Accuracy

## 📈 Results Comparison

| Metric | EfficientNet-B0 | Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | **95.75%** | 58.2% |
| **F1-Score** | **0.96** | 0.58 |
| **AUC-ROC** | **0.99** | 0.75 |

*Note: The Deep Learning approach significantly outperformed the classical ML approach due to its ability to capture complex spatial features in the ultrasound images.*

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Breast-Ultrasound-Classification.git](https://github.com/YOUR_USERNAME/Breast-Ultrasound-Classification.git)
   cd Breast-Ultrasound-Classification

   
