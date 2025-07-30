import os
import sys

#  Append project root for clean modular imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in globals() else ".", ".."))
sys.path.append(project_root)

from src.api.models import load_model
from src.api.data_loader import load_test_data
from src.api.shap_runner import run_shap

def test_shap_output_format():
    scope = "ecom"
    model = load_model(scope, "Logistic Regression")
    X, _ = load_test_data(scope)
    
    shap_values = run_shap(model, X.head(10))

    # Handle case where shap_values is a list (e.g., multiclass)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    assert hasattr(shap_values, "values")
    assert shap_values.values.shape[0] ==10