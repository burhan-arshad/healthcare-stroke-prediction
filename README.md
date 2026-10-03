# Stroke Prediction using Artificial Neural Network

An educational machine learning project that uses an Artificial Neural Network (ANN) to predict stroke risk from healthcare-related patient data. The project includes preprocessing, class-imbalance handling, model training, evaluation, and an interactive Streamlit application.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python\&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow\&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit\&logoColor=white)

---

## Overview

Stroke prediction is a binary classification problem where the model estimates whether a patient record is associated with a stroke.

The project demonstrates an end-to-end deep learning workflow:

```text
Healthcare Dataset
       ↓
Data Preprocessing
       ↓
Missing Value Handling
       ↓
Categorical Encoding
       ↓
Feature Scaling
       ↓
Class Weighting
       ↓
Artificial Neural Network
       ↓
Model Evaluation
       ↓
Streamlit Prediction App
```

The model uses patient attributes such as age, hypertension, heart disease, glucose level, BMI, smoking status, and other demographic and lifestyle features.

---

## Features

* Artificial Neural Network using TensorFlow/Keras
* Missing BMI value handling
* Categorical feature encoding
* Numerical feature scaling
* Class imbalance handling using class weights
* Binary stroke-risk classification
* Saved model and preprocessing artifacts
* Interactive Streamlit interface
* Deployment-ready project structure

---

## Dataset

The project uses the **Healthcare Stroke Dataset**, which contains patient information including:

* Gender
* Age
* Hypertension
* Heart disease
* Ever married
* Work type
* Residence type
* Average glucose level
* BMI
* Smoking status

### Target

| Value | Meaning   |
| ----- | --------- |
| `0`   | No stroke |
| `1`   | Stroke    |

The dataset is highly imbalanced, with stroke cases representing a small portion of the total records. Therefore, class weighting is used during model training rather than relying on accuracy alone.

---

## Model Architecture

The ANN is implemented using TensorFlow/Keras:

```text
Input Features
      ↓
Dense(32, ReLU)
      ↓
Dropout
      ↓
Dense(16, ReLU)
      ↓
Dropout
      ↓
Dense(1, Sigmoid)
      ↓
Stroke Prediction
```

### Training Configuration

* Optimizer: Adam
* Loss: Binary Cross-Entropy
* Output activation: Sigmoid
* Class weighting: Enabled
* Regularization: Dropout

The sigmoid output represents the model's estimated probability for the positive class.

---

## Data Preprocessing

The preprocessing pipeline includes:

1. Handling missing BMI values
2. Encoding categorical variables
3. Separating features and target
4. Scaling numerical features using `StandardScaler`
5. Preserving the feature-column order used during training
6. Applying class weights to address class imbalance

The fitted preprocessing objects are saved so that the Streamlit application applies the same transformations used during training.

---

## Saved Model Files

| File                  | Purpose                          |
| --------------------- | -------------------------------- |
| `model.keras`         | Trained ANN model                |
| `scaler.pkl`          | Fitted `StandardScaler`          |
| `feature_columns.pkl` | Feature names and training order |

Keeping the feature order consistent is important because the trained neural network expects inputs in the same structure used during training.

---

## Streamlit Application

The project includes an interactive Streamlit frontend where users can enter patient information and obtain a model prediction.

The application loads:

* Trained ANN model
* Fitted scaler
* Feature-column configuration

It then preprocesses the provided input and passes it to the trained model.

```text
User Input
    ↓
Preprocessing
    ↓
Feature Scaling
    ↓
ANN Model
    ↓
Prediction Probability
    ↓
Classification Result
```

---

## Tech Stack

| Category            | Technology         |
| ------------------- | ------------------ |
| Language            | Python             |
| Deep Learning       | TensorFlow / Keras |
| Data Processing     | Pandas, NumPy      |
| Machine Learning    | Scikit-learn       |
| Model Serialization | Joblib             |
| Web Interface       | Streamlit          |

---

## Project Structure

```text
healthcare-stroke-prediction/
│
├── app.py
├── model.keras
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/burhan-arshad/healthcare-stroke-prediction.git
cd healthcare-stroke-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser at the local Streamlit URL.

---

## Model Evaluation

Because stroke datasets are typically highly imbalanced, evaluation should consider more than accuracy.

Useful classification metrics include:

* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC

For a medical-risk classification problem, false negatives and false positives can have different implications, so metric selection should depend on the intended application.

---

## Limitations

This project is intended for **educational and demonstration purposes**.

* The dataset may not represent all populations or clinical settings.
* The model is trained on historical dataset patterns and may not generalize to new populations.
* The available features are limited compared with real clinical decision-making.
* The Streamlit application is a demonstration interface, not a clinical system.
* Model predictions should not be interpreted as medical diagnoses.

---

## Future Improvements

Potential improvements include:

* Hyperparameter optimization
* Cross-validation
* ROC-AUC and Precision-Recall analysis
* Threshold optimization
* Comparison with Random Forest, XGBoost, and other classifiers
* Explainability using SHAP or feature importance techniques
* Improved class-imbalance strategies such as SMOTE
* Model performance monitoring
* More robust validation on external datasets

---

## Learning Objectives

This project demonstrates practical implementation of:

* Artificial Neural Networks
* Binary Classification
* Healthcare Machine Learning
* Data Preprocessing
* Categorical Encoding
* Feature Scaling
* Class Imbalance
* Class Weighting
* Dropout Regularization
* TensorFlow/Keras
* Model Serialization
* Streamlit Deployment

---

## Author

**Burhan Arshad**

Computer Science Student | AI & Machine Learning

* GitHub: [@burhan-arshad](https://github.com/burhan-arshad)
* LinkedIn: [burhan-arshad](https://www.linkedin.com/in/burhan-arshad/)

---

## Disclaimer

This project is for educational and demonstration purposes only. It is **not a medical diagnostic tool** and should not be used to make medical decisions or replace professional medical advice.
