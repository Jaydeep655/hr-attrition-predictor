"""
HR Analytics — Employee Attrition Predictor
--------------------------------------------
Trains a classifier on the classic IBM HR Attrition dataset (or any CSV with a
similar schema) and serves it as an interactive Streamlit app so a recruiter
can plug in employee attributes and see a live retention-risk prediction.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy: push this folder to GitHub -> share.streamlit.io -> connect repo -> deploy.
Dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
(download `WA_Fn-UseC_-HR-Employee-Attrition.csv`, place in the same folder as `data.csv`)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os

st.set_page_config(page_title="HR Attrition Predictor", layout="wide")

MODEL_PATH = "attrition_model.joblib"
DATA_PATH = "data.csv"


@st.cache_resource
def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
                 errors="ignore")

    target = "Attrition"
    y = df[target].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[target])

    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "report": classification_report(y_test, preds, output_dict=True),
    }

    bundle = {"model": model, "encoders": encoders, "columns": list(X.columns), "metrics": metrics}
    joblib.dump(bundle, MODEL_PATH)
    return bundle


st.title("🧑‍💼 HR Attrition Risk Predictor")
st.caption(
    "Predicts employee attrition risk using a Random Forest classifier trained on the "
    "IBM HR Analytics dataset. Adjust the inputs on the left to see risk update live."
)

if not os.path.exists(DATA_PATH):
    st.warning(
        f"`{DATA_PATH}` not found. Download the IBM HR Attrition dataset from Kaggle, "
        f"save it as `{DATA_PATH}` in this folder, then reload the app."
    )
    st.stop()

bundle = train_or_load_model()
model, encoders, columns, metrics = (
    bundle["model"], bundle["encoders"], bundle["columns"], bundle["metrics"]
)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model performance")
    st.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.metric("Recall (Attrition=Yes)", f"{metrics['report']['1']['recall']*100:.1f}%")

    st.subheader("Try a prediction")
    df_raw = pd.read_csv(DATA_PATH)
    inputs = {}
    for col in columns:
        if col in encoders:
            options = list(encoders[col].classes_)
            choice = st.selectbox(col, options)
            inputs[col] = encoders[col].transform([choice])[0]
        else:
            default = int(df_raw[col].median())
            inputs[col] = st.slider(
                col, int(df_raw[col].min()), int(df_raw[col].max()), default
            )

    if st.button("Predict attrition risk", type="primary"):
        X_input = pd.DataFrame([inputs])[columns]
        proba = model.predict_proba(X_input)[0, 1]
        st.session_state["last_proba"] = proba

with col2:
    st.subheader("Feature importance")
    importances = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False).head(15)
    st.bar_chart(importances)

    if "last_proba" in st.session_state:
        proba = st.session_state["last_proba"]
        st.subheader("Prediction result")
        risk_label = "High Risk" if proba > 0.5 else "Low Risk"
        st.metric("Attrition probability", f"{proba*100:.1f}%", risk_label)
        st.progress(min(proba, 1.0))
