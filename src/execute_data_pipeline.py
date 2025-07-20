import os
import sys

# Append project root to path for clean imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.data_processing import (
    prepare_ecommerce_data,
    prepare_bank_data,
    split_and_balance
)

def run_pipeline():
    # Define paths
    path_ecom = os.path.join("data", "raw", "Fraud_Data.csv")
    path_ipmap = os.path.join("data", "raw", "IpAddress_to_Country.csv")
    path_bank = os.path.join("data", "raw", "creditcard.csv")

    # Preprocess datasets
    ecommerce_df = prepare_ecommerce_data(path_ecom, path_ipmap)
    bank_df = prepare_bank_data(path_bank)

    # Split and balance
    split_and_balance(ecommerce_df, "class", "ecom")
    split_and_balance(bank_df, "Class", "bank")

    print("✅ Task-1 pipeline executed. All cleaned and split files saved to 'data/processed/'.")

if __name__ == "__main__":
    run_pipeline()
