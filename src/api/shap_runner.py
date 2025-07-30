import shap

def run_shap(model, X):
    explainer = shap.Explainer(model, X)
    return explainer(X)
