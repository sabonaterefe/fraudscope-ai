import os
import sys
import argparse
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in globals() else ".", ".."))
sys.path.append(project_root)

from src.model_utils import evaluate_model, save_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_data(scope):
    base = os.path.join(project_root, "data", "processed")
    try:
        X_train = pd.read_csv(os.path.join(base, f"{scope}_X_train.csv"))
        X_test = pd.read_csv(os.path.join(base, f"{scope}_X_test.csv"))
        y_train = pd.read_csv(os.path.join(base, f"{scope}_y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(base, f"{scope}_y_test.csv")).values.ravel()
        logging.info(f"📥 Loaded data splits for '{scope}'")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"❌ Failed to load data for scope='{scope}': {e}")
        raise RuntimeError(f"Data loading error for scope='{scope}'")

def train_models(scope):
    logging.info(f"\n🔬 Starting model training for scope: '{scope}'")
    try:
        X_train, X_test, y_train, y_test = load_data(scope)
    except RuntimeError:
        return {}

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=1)
    }

    results = {}
    for name, model in models.items():
        try:
            logging.info(f"⚙️ Fitting model: {name}")
            model.fit(X_train, y_train)

            logging.info(f"📊 Evaluating model: {name}")
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_prob)

            results[name] = metrics
            save_model(model, name, scope)
            logging.info(f"💾 Saved model: {name} → scope='{scope}'")

        except Exception as e:
            logging.error(f"❌ Failed processing for model '{name}': {e}")

    logging.info(f"✅ Completed training for scope='{scope}'\n")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--scope", type=str, required=True, choices=["ecom", "bank"])
    args = parser.parse_args()

    metrics = train_models(args.scope)
    if metrics:
        for model, result in metrics.items():
            print(f"\n🔹 {model}")
            print("F1-Score:", result.get("F1-Score"))
            print("AUC-PR:", result.get("AUC-PR"))
            print("Confusion Matrix:\n", result.get("Confusion Matrix"))
    else:
        print(f"🚫 No metrics returned for scope='{args.scope}'")
