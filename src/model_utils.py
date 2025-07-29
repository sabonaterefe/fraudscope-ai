import os
import joblib
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, y_pred)
    return {
        "F1-Score": round(f1, 4),
        "AUC-PR": round(auc_pr, 4),
        "Confusion Matrix": cm
    }

def save_model(model, model_name, scope):
    path = os.path.join("artifacts", scope, f"{model_name}.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Model saved: {path}")
