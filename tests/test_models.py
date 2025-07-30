import pandas as pd
import os
import sys

# Append project root for clean modular imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in globals() else ".", ".."))
sys.path.append(project_root)

from src.api.models import load_model, predict

def test_model_loading_and_prediction():
    scope = "ecom"
    model_name = "XGBoost"
    model = load_model(scope, model_name)
    
    # Dummy input (should match your feature shape)
    X_sample = pd.read_csv(f"data/processed/{scope}_X_test.csv").head(5)
    
    y_pred, y_prob = predict(model, X_sample)
    
    assert len(y_pred) == len(X_sample)
    assert all(0.0 <= p <= 1.0 for p in y_prob)
