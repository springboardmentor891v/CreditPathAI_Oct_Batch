# CreditPathAI

## CreditPathAI—Automating and optimizing the loan recovery lifecycle by modeling repayment behavior using diverse data

CreditPathAI is an end-to-end machine learning system for **credit default risk prediction**, built with a **clean, production-oriented architecture**. It demonstrates how real-world ML systems are designed, structured, evaluated, and deployed — not just how models are trained.

The project focuses on **reusability, inference safety, dataset isolation, and deployment readiness**, mirroring industry ML engineering practices.

---

## Overview

Financial institutions must identify high-risk borrowers **before default occurs** to reduce losses and improve lending decisions.

CreditPathAI addresses this problem by building predictive models that estimate the probability of loan repayment failure using structured financial and borrower data. The system supports **multiple datasets** through a shared ML engine, allowing consistent experimentation, evaluation, and deployment across different data sources.


---

## System Architecture

The system follows a modular pipeline design:

* Dataset-specific ingestion and artifact storage
* Shared preprocessing, modeling, evaluation, and inference logic
* Strict separation between training and inference stages


## Pipeline Flow

```text
Raw Data
   ↓
Ingestion
   ↓
Preprocessing
   ↓
Model Training
   ↓
Evaluation
   ↓
Saved Artifacts
   ↓
Inference / Streamlit App



---

## Supported Datasets

### Kaggle Loan Default Dataset

* **Source:** Kaggle
* **Target Variable:** `repay_fail` (0 = No Default, 1 = Default)
* **Data Type:** Tabular borrower and financial features

### Microsoft Loan Dataset

* **Source:** Microsoft sample loan data
* **Target Variable:** Default indicator
* **Data Type:** Relational borrower and loan tables

Each dataset is processed independently while reusing the same core ML pipeline to ensure consistency and prevent data leakage.

---

## Machine Learning Pipeline

### Ingestion

* Load raw CSV / TXT files
* Merge relational tables when required
* Validate schema consistency

### Preprocessing

* Missing value handling
* Categorical encoding
* Feature scaling
* Feature order locking for inference safety

### Modeling

Supported models include:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Support Vector Machine (SVM)

### Evaluation

* Accuracy, Precision, Recall, F1-score
* ROC-AUC and PR-AUC
* Confusion Matrix
* Feature importance for tree-based models
* Multi-model performance comparison

### Inference

* Predictions using saved preprocessing artifacts
* Strict feature alignment enforcement
* Serialized, production-ready models

### Deployment

* Interactive Streamlit application for prediction and visualization

---

## Installation (Users)

These instructions are for users who want to **run the pipeline or test the application locally**.

```bash
pip install -r requirements.txt
```

Run the ML pipeline:

```bash
python main.py
```

Launch the Streamlit application:

```bash
streamlit run streamlit/streamlit_app.py
```

---

## Results Summary

* Tree-based models outperform linear models on nonlinear credit patterns
* Class imbalance handled using SMOTE where appropriate
* ROC-AUC used as the primary metric due to skewed class distribution
* Feature importance analysis highlights key financial risk drivers

Detailed results are available in dataset-specific README files.

---

## Tech Stack

* Python
* Pandas, NumPy
* scikit-learn
* imbalanced-learn
* XGBoost
* Matplotlib, Seaborn
* Joblib
* Streamlit

---
