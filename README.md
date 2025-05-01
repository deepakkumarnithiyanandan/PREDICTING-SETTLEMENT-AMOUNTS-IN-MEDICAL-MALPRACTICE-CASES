# PREDICTING-SETTLEMENT-AMOUNTS-IN-MEDICAL-MALPRACTICE-CASES

## 🧠 Overview

This project aims to predict **settlement amounts in medical malpractice cases** using supervised machine learning techniques. Settlement prediction is a critical task for insurance companies, legal teams, and healthcare providers to estimate potential liability and make informed decisions.

The project covers an end-to-end machine learning workflow—from **exploratory data analysis (EDA)** and **data preprocessing**, to **model training**, **evaluation**, and **explainability**. The focus is on building not only an accurate predictive model, but one that is interpretable and adaptable to real-world complexity.

## 🎯 Objectives

- Identify key features influencing settlement outcomes.
- Build multiple regression models to estimate settlement amounts.
- Apply interpretability tools (e.g., SHAP) to understand model behavior.
- Evaluate performance using meaningful regression metrics.

## 📊 Dataset

The dataset includes anonymized records of medical malpractice claims, consisting of:
- Patient/provider demographics
- Injury type and severity
- Case outcome variables (e.g., settlement, trial verdict)

## ⚙️ What This Project Does

- Conducts EDA using **Pandas**, **Matplotlib**, and **Seaborn**
- Applies preprocessing techniques like **label encoding**, **feature scaling**, and **feature selection**
- Trains models including:
  - **Random Forest**
  - **LightGBM**
  - **TabNet**
  - **Neural Network (MLP)**
- Tunes models using **RandomizedSearchCV** with **cross-validation**
- Evaluates performance using **RMSE** and **R² score**
- Uses **SHAP** for model interpretability and feature impact analysis
- Saves trained models using **joblib** for deployment

✅ **Achieved an R² score of 70%** despite significant data imbalance, demonstrating model robustness and generalizability.

## 🧰 Tech Stack

Python, Scikit-learn, LightGBM, TabNet, SHAP, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook
