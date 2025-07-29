## FraudScope AI — Task 1: Data Preprocessing & Class Imbalance Handling
## This repository contains the first stage of the FraudScope AI pipeline, which prepares raw banking and e-commerce datasets for fraud detection. Task 1 covers data cleaning, geolocation enrichment, feature engineering, class imbalance handling, and exploratory data analysis.

#  Datasets
Managed via DVC and hosted on DagsHub. Not tracked via Git.

Fraud_Data.csv: E-commerce transaction metadata (purchase_time, device_id, source, browser, age, ip_address, etc.)

IpAddress_to_Country.csv: Maps IP ranges to countries

creditcard.csv: PCA-transformed bank transactions with extreme class imbalance

All datasets contain highly imbalanced targets (class) indicating fraud (1) or non-fraud (0).

# Access the datasets directly on DagsHub’s data tab at : https://dagshub.com/sabonaterefe/fraudscope-ai?filter=data

# Task 1 Objectives
Handle missing and duplicate entries

Engineer temporal and behavioral features from transaction metadata

Map IPs to countries using IP range matching

Scale numeric features and encode categorical ones

Split datasets and apply SMOTE to mitigate imbalance

Save reproducible train-test splits for modeling

# Repository Structure
src/data_preprocessing.py: Contains full preprocessing logic for both datasets, including cleaning, merging, feature engineering, encoding, and balancing

src/execute_data_pipeline.py: Orchestrates execution of the preprocessing pipeline

notebooks/eda_fraudscope.ipynb: Provides univariate and bivariate analysis to understand feature distributions and fraud patterns

requirements.txt: Dependency list for all preprocessing and analysis steps

All processed datasets and splits are saved in data/processed/ (ignored by Git).

#  Learning Outcomes
Skills

Implementing robust data pipelines for fraud detection

Handling extreme class imbalance with SMOTE

Engineering time-aware behavioral features

Encoding and scaling features for classification

Knowledge

Business-centric thinking around fraud detection

IP range matching and geolocation enrichment

Justifying preprocessing decisions based on domain insights

Behaviors

Systematic, reproducible pipeline development

Maintaining separation of concerns across modular scripts

Reporting insights clearly via EDA

# Pipeline Summary
E-commerce

Timestamp parsing: signup_time, purchase_time

Feature engineering: time_since_signup, hour_of_day, day_of_week, transaction velocity

IP enrichment: convert ip_address to integer and map to country

Encoding: one-hot for categorical variables

Class imbalance: SMOTE applied after train-test split

Bank

Data cleaning and duplicate removal

Scaling: Amount standardized

Class imbalance: SMOTE applied after split

#  How to Run
Clone the repo from DagsHub

Set up Python environment using requirements.txt

Execute preprocessing pipeline via:

python src/execute_data_pipeline.py

View cleaned datasets in data/processed/

Open eda_fraudscope.ipynb for visual exploration


 Task 2 — Model Training and Evaluation
This phase builds on the feature engineering pipeline to implement reliable fraud classification using logistic regression and XGBoost. It emphasizes repeatability, metric-driven selection, and clear separation of logic for maintainability.
🔧 Modules
- src/model_training.py — Defines model training routines
- src/model_utils.py — Handles metric computation and serialization
- src/extended_pipeline_task2.py — End-to-end pipeline execution
- notebooks/evaluation_report.ipynb — Visual comparison of classifier performance
📊 Metrics Used
- F1-Score — Captures balance between precision and recall
- AUC-PR — Optimized for imbalance sensitivity
- Confusion Matrix — Highlights misclassifications
▶️ Execution
python src/extended_pipeline_task2.py


Model artifacts and evaluation results are stored under:
- artifacts/{ecom, bank}/
- models/
Evaluation reports are accessible through the notebook:
jupyter notebook notebooks/evaluation_report.ipynb



📌 Task 3 — Model Explainability with SHAP
This component introduces post-hoc interpretability using SHAP to deconstruct model predictions and support auditability. It delivers local and global insights for transparency across business and regulatory boundaries.
🔧 Modules
- src/explainability.py — Generates SHAP plots
- notebooks/shap_explainability.ipynb — Visual dashboard for interpretation
📊 Outputs
- SHAP Summary Plot — Global feature contribution
- SHAP Force Plot — Individual prediction breakdown
▶️ Execution
jupyter notebook notebooks/shap_explainability.ipynb


Plots are saved to:
- models/{ecom, bank}_shap_summary_plot.png
- models/{ecom, bank}_shap_force_plot.png

This documentation concludes the core modeling and interpretability stages for FraudScope AI. Each module is designed for modular reuse, transparent evaluation, and stakeholder alignment.


## Maintained by Sabona Terefe — Machine Learning Engineer specializing in NLP, modular pipelines, and scalable data infrastructure.
