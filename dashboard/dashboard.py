import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import os
from sklearn.metrics import f1_score, roc_auc_score

# 🔧 Configs
SCOPES = ["ecom", "bank"]
ARTIFACT_DIR = "artifacts"
DATA_DIR = "data/processed"

# 🌐 Page Setup
st.set_page_config(page_title="FraudScope AI", layout="wide", page_icon="🔍")

# 🎨 Custom Styling
st.markdown("""
    <style>
    .metric-label { font-size: 1.1rem; font-weight: 600; }
    .block-container { padding-top: 2rem; }
    .stDataFrame { font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

# 🔄 Initialize Session State
if "scope" not in st.session_state:
    st.session_state.scope = SCOPES[0]

# 🔘 Sidebar Selection
new_scope = st.sidebar.selectbox("Choose Data Scope", SCOPES, index=SCOPES.index(st.session_state.scope))
if new_scope != st.session_state.scope:
    st.session_state.scope = new_scope

# 📥 Load Data & Models (only if scope changes)
@st.cache_data(ttl=300, show_spinner=False)
def load_data(scope):
    X = pd.read_csv(os.path.join(DATA_DIR, f"{scope}_X_test.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, f"{scope}_y_test.csv")).squeeze()
    return X, y

@st.cache_resource(ttl=300, show_spinner=False)
def load_models(scope):
    return {
        "Logistic Regression": joblib.load(os.path.join(ARTIFACT_DIR, scope, "Logistic Regression.pkl")),
        "XGBoost": joblib.load(os.path.join(ARTIFACT_DIR, scope, "XGBoost.pkl")),
    }

X_test, y_test = load_data(st.session_state.scope)
models = load_models(st.session_state.scope)

# 🧮 Metrics Display
def display_metrics(y_true, y_pred, y_prob):
    col1, col2 = st.columns(2)
    col1.metric("🎯 F1 Score", f"{f1_score(y_true, y_pred):.3f}")
    col2.metric("📐 AUC-PR", f"{roc_auc_score(y_true, y_prob):.3f}")

# 🔎 SHAP Summary
def show_shap_summary(model, X, model_name):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    st.markdown(f"### 🔍 SHAP Summary — *{model_name}*")
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.clf()

# 🧭 Dashboard Content
st.title("📊 FraudScope AI")
st.subheader("Realtime explainability and model evaluation")

with st.expander("📁 Dataset Snapshot"):
    st.write(f"Sample from `{st.session_state.scope}_X_test.csv`")
    st.dataframe(X_test.head())

for name, model in models.items():
    st.markdown(f"## 🔹 {name} Results")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    display_metrics(y_test, y_pred, y_prob)
    show_shap_summary(model, X_test, name)
    st.divider()

st.caption("Built by Sabona T.")
