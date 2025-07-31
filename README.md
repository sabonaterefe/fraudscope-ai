## FraudScope AI
## End-to-End Fraud Detection Pipeline for E-commerce and Banking Domains Maintained by: Sabona Terefe, Machine Learning Engineer

 # Overview
FraudScope AI is a modular and scalable pipeline designed to detect fraudulent transactions across banking and e-commerce platforms. Built with a focus on transparency, reproducibility, and explainability, the system spans from raw data ingestion to SHAP-powered interpretation of model predictions.

This repository comprises three structured phases:

Task 1 — Data Preprocessing & Feature Engineering

Task 2 — Model Training & Evaluation

Task 3 — Model Interpretability using SHAP

All artifacts are version-controlled via DVC (Data Version Control) and stored on DagsHub.

Datasets
Datasets are not tracked via Git. Managed using DVC and hosted on DagsHub.

Fraud_Data.csv: E-commerce transaction metadata (timestamp, user-agent, source, etc.)

IpAddress_to_Country.csv: IP range mapping for country enrichment

creditcard.csv: PCA-based bank transactions with extreme imbalance

All datasets include a binary target class indicating fraud (1) or non-fraud (0). 📎 Access full datasets: DagsHub Data Tab

# Task 1 — Preprocessing & Class Imbalance Handling
Objectives
Clean and deduplicate transaction records

Engineer temporal, behavioral, and geolocation features

Encode categorical variables and scale numerical ones

Apply SMOTE post-split to handle class imbalance

Persist reproducible train-test splits under data/processed/

Highlights
E-commerce

Temporal signals: time_since_signup, hour_of_day, day_of_week

Behavioral signals: device_transaction_count, user_transaction_count

IP enrichment: Integer conversion and country mapping

Encoding: One-hot for categorical, standard scaling for numerical

Banking

Cleanup and scaling of raw PCA components

Class imbalance addressed via SMOTE after splitting

Files
src/data_preprocessing.py: Modular preprocessing for both domains

src/execute_data_pipeline.py: Entrypoint for Task 1 pipeline

notebooks/eda_fraudscope.ipynb: Visual EDA on feature distributions and fraud clusters

requirements.txt: Dependencies for full pipeline execution

Run Preprocessing
python src/execute_data_pipeline.py
Explore results:

jupyter notebook notebooks/eda_fraudscope.ipynb
# Task 2 — Model Training & Evaluation
Goals
Train interpretable and high-performance models (Logistic Regression, XGBoost)

Evaluate using fraud-aware metrics

Serialize models and evaluation artifacts for audit and reuse

Evaluation Metrics
F1-Score: Precision-recall balance

AUC-PR: Preferred under severe class imbalance

Confusion Matrix: Error profiling for fraud vs non-fraud

Files
src/model_training.py: Training routines for each model

src/model_utils.py: Metric computation and model saving

src/extended_pipeline_task2.py: Full pipeline executor

notebooks/evaluation_report.ipynb: Visual comparison of model performance

# Run Training
python src/extended_pipeline_task2.py
View results:

jupyter notebook notebooks/evaluation_report.ipynb
Outputs Saved to
artifacts/ecom/, artifacts/bank/

models/ → includes trained .pkl files and metric snapshots

# Task 3 — Model Interpretability with SHAP
Transparency is non-negotiable in fraud modeling. SHAP (SHapley Additive exPlanations) was integrated to:

Clarify global feature importance

Unpack individual fraud predictions

Support trust, audit, and debugging efforts

Visual Outputs
Summary Plot: Mean feature impact across dataset

Force Plot: Local explanation for individual transactions

Waterfall Plot: Step-by-step contribution breakdown

Files
src/explainability.py: SHAP explainer logic

notebooks/shap_explainability.ipynb: Dashboard for interpretation

# Run SHAP Analysis
jupyter notebook notebooks/shap_explainability.ipynb
Plots Saved To
models/{ecom, bank}_shap_summary_plot.png

models/{ecom, bank}_shap_force_plot.png

models/{ecom, bank}_shap_waterfall_plot.png

# Learning Outcomes
Skills Developed

Modular pipeline engineering

Imbalanced classification using SMOTE

Feature engineering for fraud behavior and temporal signals

XGBoost tuning and performance profiling

SHAP-based model interpretation

Core Concepts Applied

IP-based geolocation mapping

PCA feature handling for anonymized banking data

Post-split resampling to prevent data leakage

Explaining fraud decisions using SHAP visuals

Engineering Practices

Version-controlled artifacts with DVC

Separation of concerns across tasks

Clear reporting via notebooks

Audit-ready model metadata

# Getting Started
Clone the repo from DagsHub

Create a virtual environment and install dependencies:

pip install -r requirements.txt
Execute preprocessing, training, and interpretation workflows

Explore notebooks for visual insights

# Maintainer
# Sabona Terefe Machine Learning Engineer Specialized in NLP, scalable pipelines, and structured fraud systems
