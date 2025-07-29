import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.model_utils import evaluate_model, save_model

def train_models(scope):
    print(f"\n🔬 Training models for '{scope}' dataset")

    # Load preprocessed split data
    path_X_train = os.path.join("data", "processed", f"{scope}_X_train.csv")
    path_X_test = os.path.join("data", "processed", f"{scope}_X_test.csv")
    path_y_train = os.path.join("data", "processed", f"{scope}_y_train.csv")
    path_y_test = os.path.join("data", "processed", f"{scope}_y_test.csv")

    X_train = pd.read_csv(path_X_train)
    X_test = pd.read_csv(path_X_test)
    y_train = pd.read_csv(path_y_train).values.ravel()
    y_test = pd.read_csv(path_y_test).values.ravel()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=1)
    }

    metrics_dump = {}
    for name, model in models.items():
        print(f"⚙️ Training {name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        metrics_dump[name] = metrics

        save_model(model, name, scope)

    print(f"✅ Completed training for '{scope}'")
    return metrics_dump
