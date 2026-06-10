
# Disease Prediction from Medical Data

## Overview

This project uses Machine Learning techniques to predict whether a breast tumor is **Benign** or **Malignant** using the Breast Cancer Wisconsin Dataset.

The project demonstrates a complete machine learning workflow including:

* Data Preprocessing
* Feature Scaling
* Model Training
* Model Evaluation
* Model Comparison
* Performance Visualization

Three classification algorithms are trained and compared:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest

---

# Problem Statement

Early detection of breast cancer is crucial for improving treatment outcomes and reducing mortality rates.

The objective of this project is to develop machine learning models capable of predicting breast cancer diagnosis based on tumor characteristics extracted from medical data.

---

# Dataset

Dataset: Breast Cancer Wisconsin Dataset

### Features

Examples of input features:

* Radius Mean
* Texture Mean
* Perimeter Mean
* Area Mean
* Smoothness Mean
* Compactness Mean
* Concavity Mean
* Symmetry Mean
* Fractal Dimension Mean

### Target Variable

| Value | Diagnosis |
| ----- | --------- |
| 0     | Benign    |
| 1     | Malignant |

Dataset Size:

* 569 Samples
* 30 Numerical Features

---

# Project Structure

```text
Disease-Prediction-from-Medical-Data/
│
├── data/
│   ├── data.csv
│   └── processed/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   └── random_forest.pkl
│
├── outputs/
│   ├── evaluation_summary.csv
│   ├── roc_curve_comparison.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_svm.png
│   └── confusion_matrix_random_forest.png
│
├── notebook/
│   └── Breast_Cancer_EDA.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Machine Learning Pipeline

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Label Encoding
   │
   ▼
Train-Test Split
   │
   ▼
Feature Scaling
   │
   ▼
Model Training
   │
   ├── Logistic Regression
   ├── SVM
   └── Random Forest
   │
   ▼
Model Evaluation
   │
   ├── Accuracy
   ├── Precision
   ├── Recall
   ├── F1 Score
   └── ROC-AUC
   │
   ▼
Model Comparison
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Pickle

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Disease-Prediction-from-Medical-Data.git

cd Disease-Prediction-from-Medical-Data
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Training Pipeline

Train all models:

```bash
python main.py
```

Expected Output:

```text
Step 1: Data Preprocessing

Step 2: Training Models

Saved logistic_regression.pkl
Saved svm.pkl
Saved random_forest.pkl

Training Completed
```

---

# Run Model Evaluation

```bash
python src/evaluate.py
```

Outputs Generated:

* Evaluation Summary CSV
* ROC Curve Comparison
* Confusion Matrix for each model

---

# Evaluation Metrics

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

These metrics help compare model performance and identify the best-performing classifier.

---

# Model Comparison

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~97%     |
| SVM                 | ~98%     |
| Random Forest       | ~97%     |

*Actual results may vary slightly depending on train-test split.*

---

# Future Improvements

* Hyperparameter Tuning using GridSearchCV
* XGBoost Integration
* SHAP Explainability
* Feature Importance Dashboard
* Streamlit Web Application
* Docker Deployment
* MLOps Pipeline

---

# Learning Outcomes

Through this project, I gained practical experience in:

* Data Preprocessing
* Classification Algorithms
* Feature Scaling
* Model Evaluation
* Medical Data Analysis
* Machine Learning Project Structure

---

# Author

**Hardik Yerne**

B.Tech Computer Science Student

Interested in Machine Learning, Deep Learning, NLP, LLMs, MLOps, and AI Applications.
