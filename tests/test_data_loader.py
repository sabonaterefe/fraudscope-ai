import os
import sys

# Append project root for clean modular imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in globals() else ".", ".."))
sys.path.append(project_root)

from src.api.data_loader import load_test_data

def test_data_shape_and_labels():
    scope = "bank"
    X, y = load_test_data(scope)
    
    assert X.shape[0] == len(y)
    assert X.shape[1] > 0  # At least one feature
    assert set(y.unique()).issubset({0, 1})
