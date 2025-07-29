import os
import sys

# 📦 Append project root for clean modular imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 🔧 Import pipeline components
from src.data_processing import (
    prepare_ecommerce_data,
    prepare_bank_data,
    split_and_balance
)
from src.model_training import train_models  # Task 2

def run_full_pipeline():
    print("\n🛠️ Initializing pipeline...\n")

    # 📁 Define data paths
    path_ecom = os.path.join("data", "raw", "Fraud_Data.csv")
    path_ipmap = os.path.join("data", "raw", "IpAddress_to_Country.csv")
    path_bank = os.path.join("data", "raw", "creditcard.csv")

    # 🧹 Task 1: Preprocessing
    print("🔄 Preprocessing E-commerce Dataset")
    ecommerce_df = prepare_ecommerce_data(path_ecom, path_ipmap)

    print("🔄 Preprocessing Credit Card Dataset")
    bank_df = prepare_bank_data(path_bank)

    print("⚖️ Balancing E-commerce Dataset")
    split_and_balance(ecommerce_df, "class", "ecom")

    print("⚖️ Balancing Credit Card Dataset")
    split_and_balance(bank_df, "Class", "bank")

    print("✅ Task 1 Complete: Preprocessing & Balancing\n")

    # 🤖 Task 2: Model Training & Evaluation
    print("🚀 Task 2: Training Models\n")

    fraud_metrics = train_models("ecom")
    credit_metrics = train_models("bank")

    # 📊 Display metrics — Fraud Dataset
    print("\n📊 Results: Fraud Dataset")
    for model, result in fraud_metrics.items():
        print(f"\n🔹 {model}")
        print("F1-Score:", result.get('F1-Score'))
        print("AUC-PR:", result.get('AUC-PR'))
        print("Confusion Matrix:\n", result.get('Confusion Matrix'))

    # 📊 Display metrics — Credit Card Dataset
    print("\n📊 Results: Credit Card Dataset")
    for model, result in credit_metrics.items():
        print(f"\n🔹 {model}")
        print("F1-Score:", result.get('F1-Score'))
        print("AUC-PR:", result.get('AUC-PR'))
        print("Confusion Matrix:\n", result.get('Confusion Matrix'))

    print("\n✅ Pipeline execution complete.")

if __name__ == "__main__":
    run_full_pipeline()
