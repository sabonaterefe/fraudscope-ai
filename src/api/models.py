import joblib
import os

def load_model(scope, model_name):
    return joblib.load(os.path.join("artifacts", scope, f"{model_name}.pkl"))

def predict(model, X):
    return model.predict(X), model.predict_proba(X)[:, 1]
