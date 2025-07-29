import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

def explain_model(prefix, model_type="xgb"):
    model_path = f"models/{prefix}_{model_type}.pkl"
    model = joblib.load(model_path)
    X_test = pd.read_csv(f"data/processed/{prefix}_X_test.csv")

    # Use TreeExplainer for tree-based models to access expected_value
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Global summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f"models/{prefix}_summary_plot.png")
    plt.close()

    # Local force plot for first instance
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=True)
    plt.savefig(f"models/{prefix}_force_plot.png")