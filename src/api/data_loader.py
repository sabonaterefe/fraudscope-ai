import pandas as pd
import os

def load_test_data(scope):
    X = pd.read_csv(os.path.join("data", "processed", f"{scope}_X_test.csv"))
    y = pd.read_csv(os.path.join("data", "processed", f"{scope}_y_test.csv")).squeeze()
    return X, y
