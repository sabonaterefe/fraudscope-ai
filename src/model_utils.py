import os
import joblib
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix, auc

def save_model(model, model_name, scope):
    artifact_dir = os.path.join("artifacts", scope)
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, f"{model_name}.pkl")
    joblib.dump(model, path)

def evaluate_model(y_true, y_pred, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "F1-Score": f1_score(y_true, y_pred),
        "AUC-PR": auc(recall, precision),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }
